"""al_oracle_curve_diag.py: the ORACLE ACQUISITION CURVE -- does the label budget
itself, or the acquisition algorithm, bound the few-label gain? (new_iters.md,
the highest-value remaining experiment.)

We have only ever measured oracle-U (the direction) with STATIC acquisition
(Iteration 1: +0.29-0.37 at b=8). This asks the complementary question: given the
RIGHT POINTS, how much does b buy? It separates the three stories:

  Story A  good AL + few labels -> huge gain          (optimize acquisition)
  Story B  gradual 8->16->32 curve                    (label-efficiency problem)
  Story C  even oracle AL doesn't move until hundreds  (label-starved, not
           algorithm-starved)

Design: FIX the downstream (the oracle-U first-order step W1 = W0 + rho*U_or*
G/||G||, the ONLY few-label mechanism that works -- Iteration 1), and vary ONLY
the point SELECTION. This isolates the acquisition question from the update
question (updates are closed; we measure point value under the one working
update).

Acquisition arms (all label-free except the oracle arms):
  random             fixed-seed random baseline
  margin_tta_div     the Iteration-1 one-shot winner (the realistic AL reference)
  oracle_error       query the pool points the FROZEN probe gets wrong (true
                     labels used for SELECTION only -- the "oracle error AL"
                     ceiling)
  oracle_pair        query the boundary points of the VAL-TRUTH top error pairs
                     (true pairs used for SELECTION only)
  margin_perm        margin_tta_div selection but with LABELS PERMUTED in the
                     downstream step -- the control: if true labels do not beat
                     permuted labels, the acquisition extracts no supervised
                     structure

Budgets: 2,4,8,16,32,64 (the phase-transition sweep).

Also reported per condition:
  gain_concentration: fraction of VAL points whose prediction changes under the
      ORACLE decoder (W*) vs the frozen probe. This is the "does W* change 2% or
      60% of predictions" number -- if tiny, global W-adaptation was never the
      right abstraction; the problem is identifying the few changed points.

Read:
  oracle_error >> margin_tta_div  -> a real acquisition gap exists; the frozen
      probe's own errors are findable and label-valuable (a concrete target).
  oracle_error ~ margin_tta_div   -> the recoverable gain is not concentration
      on the probe's errors; the problem is not "which points to query".
  margin_perm ~ random            -> the labels' supervised content is real.
  all flat at small b until 32-64 -> Story C: label-starved.
  oracle_error pulls away early   -> Story A: acquisition is the lever.

Usage:
  uv run python robust_diagnostic/al_oracle_curve_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_oracle_curve_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11
ARMS = ['random', 'margin_tta_div', 'oracle_error', 'oracle_pair', 'margin_perm']


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'], test_sequences=None,
                  labels=data["labels"], color_map=data["color_map"], learning_map=data["learning_map"],
                  learning_map_inv=data["learning_map_inv"], sensor=arch["dataset"]["sensor"],
                  max_points=arch["dataset"]["max_points"], batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_clean(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device); labels = batch[2].to(device).view(-1); mask = (batch[1].to(device) > 0).view(-1)
            out = model(in_vol); z8 = out[2] if len(out) == 3 else out[1]
            zf = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(zf.cpu()); lbls.append(labels[mask].cpu())
    return torch.cat(feats), torch.cat(lbls)


def hdc_codes(feats, proj, device, chunk=100000):
    out = []
    for s in range(0, len(feats), chunk):
        out.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(out)


def onehot(lbls, nc):
    y = torch.zeros(len(lbls), nc); y[torch.arange(len(lbls)), lbls.long()] = 1; return y


def decode(W, codes, chunk=100000):
    W = W.detach().cpu()
    p = []
    for s in range(0, len(codes), chunk):
        p.append((codes[s:s+chunk].float() @ W).argmax(1))
    return torch.cat(p)


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device); torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1], max(1, X.shape[0] - 1))
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P; Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device); That = XP.t() @ Yd
    x0 = P @ torch.linalg.solve(Shat, That)
    if X.shape[0] <= 8:
        return x0.float()
    x = x0; b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
    r = b - A(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = A(p); d = (p * Ap).sum(0)
        if not torch.isfinite(d).all() or d.abs().max().item() < 1e-20:
            break
        a = rs / (d + 1e-30); x = x + a.unsqueeze(0) * p
        r = r - a.unsqueeze(0) * Ap; rsn = (r * r).sum(0); be = rsn / (rs + 1e-30)
        p = r + be.unsqueeze(0) * p; rs = rsn
        if not torch.isfinite(x).all():
            return x0.float()
    return x.float()


def right_topk_svd(M, r):
    _, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    if not torch.isfinite(S).all():
        S = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
        Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))
    return Vh[:r].t().float(), S.float()


def farthest_point(feats, cand_idx, b, device):
    cf = F.normalize(feats[cand_idx].float(), p=2, dim=1).to(device)
    torch.manual_seed(3)
    sel = [int(torch.randint(len(cand_idx), (1,)).item())]
    dist = (cf - cf[sel[0]]).norm(dim=1)
    for _ in range(b - 1):
        nxt = int(dist.argmax().item())
        sel.append(nxt)
        d2 = (cf - cf[nxt]).norm(dim=1)
        dist = torch.minimum(dist, d2)
    return cand_idx[torch.tensor(sel)]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync(); return time.time()


def toc(t0):
    sync(); return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=30000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--budgets", type=str, default="2,4,8,16,32,64")
    ap.add_argument("--rho_sweep", type=str, default="0.05,0.2,0.8")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--conds", type=str, default="fog,crosstalk")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    budgets = [int(x) for x in args.budgets.split(',')]
    bmax = max(budgets)
    rho_sweep = [float(x) for x in args.rho_sweep.split(',')]
    arms = [x.strip() for x in args.arms.split(',') if x.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'budgets': budgets,
               'rho_sweep': rho_sweep, 'arms': arms, 'conds': {}}

    mc = min(args.max_clean, len(fa)); ci = torch.randperm(len(fa))[:mc]
    Xc = hdc_codes(fa[ci], proj, device).float()
    W0 = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
    W0c = W0.detach().cpu()
    del Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for cond in conds:
        t0 = tic()
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        fd, ld = extract_clean(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42); perm = torch.randperm(len(fd))
        pool_f, pl = fd[perm[:args.pool_size]], ld[perm[:args.pool_size]]
        val_f, vl = fd[perm[-args.val_size:]], ld[perm[-args.val_size:]]
        Xp = hdc_codes(pool_f, proj, device).float()
        Xv = hdc_codes(val_f, proj, device).float()
        del val_f, fd, ld
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        Ws = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam, args.cg_iters, args.nystrom_m, device)
        R = (Ws - W0).detach().cpu().float()
        U_or, _ = right_topk_svd(R.t(), 2)
        U_or_g = U_or.to(device)
        W0_g = W0c.to(device)
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- gain concentration: fraction of val predictions changed by oracle ----
        pred_v_frozen = (Xv.float() @ W0c).argmax(1)
        pred_v_oracle = (Xv.float() @ Ws.cpu()).argmax(1)
        gain_conc = float((pred_v_frozen != pred_v_oracle).float().mean().item())

        # ---- frozen probe stats on the pool ----
        Lp = Xp.float() @ W0c
        pred_p = Lp.argmax(1)
        top2p = torch.topk(Lp, 2, dim=1)
        margin_p = (top2p.values[:, 0] - top2p.values[:, 1]).abs()
        a_p = top2p.indices[:, 0]; b_p = top2p.indices[:, 1]

        # ---- candidate set (boundary region) ----
        n_cand = max(int(args.cand_frac * len(Xp)), 8 * bmax)
        cand = torch.argsort(margin_p)[:n_cand]
        cand_margin = margin_p[cand]
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)

        # ---- val-truth top error pairs (for oracle_pair) ----
        # build confusion over the POOL frozen errors for the true-pair oracle
        pool_err = pred_p != pl
        true_conf = {}
        for i in torch.nonzero(pool_err).squeeze(1):
            key = (int(a_p[i]), int(b_p[i]))
            true_conf[key] = true_conf.get(key, 0) + 1
        sorted_pairs = sorted(true_conf.items(), key=lambda kv: kv[1], reverse=True)[:4]

        # ---- selectors return POOL indices ----
        def select_random(b):
            torch.manual_seed(7)
            return cand[torch.randperm(n_cand)[:b]]

        def select_margin_tta_div(b):
            score = -m + v
            topM = torch.argsort(score, descending=True)[:8 * b]
            # farthest_point returns POOL indices (cand[topM] are pool indices)
            return farthest_point(pool_f, cand[topM], b, device)

        def select_oracle_error(b):
            # the pool points the FROZEN probe gets wrong, lowest margin first
            err_idx = torch.nonzero(pool_err).squeeze(1)
            return err_idx[torch.argsort(margin_p[err_idx])[:b]]

        def select_oracle_pair(b):
            # boundary points of the val-truth top error pairs
            pair_keys = set(pair for pair, _ in sorted_pairs)
            in_pair = torch.tensor([(int(a_p[i]), int(b_p[i])) in pair_keys
                                    for i in range(len(Xp))], dtype=torch.bool)
            pair_idx = torch.nonzero(in_pair).squeeze(1)
            if len(pair_idx) == 0:
                return select_random(b)
            return pair_idx[torch.argsort(margin_p[pair_idx])[:b]]

        cond_res = {'refs': refs, 'gap': float(gap),
                    'gain_concentration': gain_conc, 'arms': {}}
        for arm in arms:
            entry = {}
            for b in budgets:
                if arm == 'random':
                    sel = select_random(b)
                elif arm == 'margin_tta_div':
                    sel = select_margin_tta_div(b)
                elif arm == 'oracle_error':
                    sel = select_oracle_error(b)
                elif arm == 'oracle_pair':
                    sel = select_oracle_pair(b)
                elif arm == 'margin_perm':
                    sel = select_margin_tta_div(b)
                else:
                    raise ValueError(arm)
                X_lab = Xp[sel].float().to(device)
                if arm == 'margin_perm':
                    # permute the labels: same points, shuffled supervision
                    torch.manual_seed(13)
                    perm_lab = pl[sel][torch.randperm(len(sel))]
                    Y_lab = onehot(perm_lab, NUM_CLASSES).float().to(device)
                else:
                    Y_lab = onehot(pl[sel], NUM_CLASSES).float().to(device)
                resid = Y_lab - X_lab @ W0_g
                G = (X_lab @ U_or_g).t() @ resid
                gcs = {}
                for rho in rho_sweep:
                    Gn = G / (G.norm() + 1e-8)
                    W1 = W0c + (U_or_g @ (rho * Gn)).cpu().float()
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs[str(rho)] = {'delta': float(d),
                                     'gap_closed': float(d / gap) if gap > 1e-9 else None}
                best = max((v['gap_closed'] or -9 for v in gcs.values()), default=None)
                entry[str(b)] = {'gc': gcs, 'best_gc': best, 'n_labels': int(len(sel))}
            cond_res['arms'][arm] = entry
            line = " ".join(f"b{b}:{v['best_gc']:+.2f}" for b, v in entry.items())
            print(f"    {arm:14s} {line}")

        results['conds'][cond] = cond_res
        del Xv, Ws, R, U_or, Lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"    gain_concentration (oracle changes this fraction of val preds): {gain_conc:.3f}")
        for arm in arms:
            e = cond_res['arms'][arm]
            line = " ".join(f"b{b}:{v['best_gc']:+.2f}" for b, v in e.items())
            print(f"    {arm:14s} {line}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Same downstream (oracle-U first-order) for every arm; only the POINT")
    print("selection varies. The decisive number is best_gc at each budget:")
    print("  oracle_error >> margin_tta_div  -> a real acquisition gap: the frozen")
    print("      probe's own errors are findable and label-valuable (a target).")
    print("  oracle_error ~ margin_tta_div   -> the gain is not concentrated on the")
    print("      probe's errors; the problem is not 'which points to query'.")
    print("  margin_perm ~ random            -> the labels' supervised content is real.")
    print("  all flat until 32-64 (Story C)  -> label-starved, not algorithm-starved.")
    print("  oracle_error pulls away early   -> Story A: acquisition is the lever.")
    print("  gain_concentration ~ 0.02       -> global W-adaptation is the wrong")
    print("      abstraction; the task is identifying the few changed points.")


if __name__ == "__main__":
    main()
