"""al_sequential_al_diag.py: A3 -- SEQUENTIAL AL (labels reveal the next query).

The Level-3 acquisition candidate (new_iters.md). Every W-update family is closed
(Iterations 3b, 4), and the pair-REPAIR loop is closed (Iteration 5). The one
remaining question for acquisition: does SEQUENTIAL acquisition -- where each
label reveals where to look NEXT -- beat one-shot acquisition?

The Iteration-1 acquisition sweep fixed the downstream (oracle-U first-order step,
the only few-label mechanism that works) and varied the acquisition RULE one-shot.
A3 extends this to the ADAPTIVE loop:

  query x1 -> see (pred,true) -> the label reveals its error pair (a,b)
  -> focus the NEXT query on the a-b boundary (low margin, top-2 = (a,b))
  -> query x2 -> reveal another pair / confirm -> focus the next boundary
  -> ... until budget b.

This directly tests "labels reveal the structure of the next query": if the
labels carry structure, sequential querying should concentrate the budget on the
recoverable boundary and beat a one-shot rule at the same budget.

Design (same downstream for every arm, only the acquisition varies):
  Downstream: the R5 first-order step W1 = W0 + rho * U_or * G/||G|| with
  G = (X_lab U_or)^T (Y - X_lab W0), U_or = top-2 of R (oracle). This is the
  ONLY few-label mechanism that closes real gc (+0.29-0.37, Iteration 1); fixing
  it isolates the acquisition question.

  Acquisition arms at b in {2,4,8}:
    random             one-shot baseline (fixed seed)
    margin_tta_div     one-shot Iteration-1 winner (the reference)
    seq_margin         sequential: always query the lowest-margin boundary point
                       (no pair focus -- does the ADAPTIVE ordering itself help?)
    seq_pair           sequential: after each label, if it revealed an error pair
                       (a,b), focus the next query on the a-b boundary; else fall
                       back to the lowest-margin point
    seq_pair_tta       seq_pair but within the focused pair use margin + TTA
                       instability to pick the point
    seq_pair_div       seq_pair but within the focused pair use farthest-point
                       diversity

Reported per budget:
  gc (best over rho) per arm -- the decisive number: does sequential beat
      margin_tta_div / random at the SAME budget?
  error-pair focus: how many of the b queries were driven by a revealed pair
      (the sequential mechanism actually engaging)
  boundary share: fraction of the b labels that are on the frozen probe's
      boundary (top-2 margin region) -- are the sequential labels landing where
      the recoverable error is?

Read:
  seq_* gc > margin_tta_div gc  -> sequential acquisition adds real value; the
      labels reveal where to look next.
  seq_* ~ margin_tta_div gc     -> the adaptive loop does not beat a good one-shot
      rule; the labels do not sharpen the next query beyond static scores.
  seq_* < random gc             -> the loop actively hurts.

Usage:
  uv run python robust_diagnostic/al_sequential_al_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_sequential_al_<label>.json
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
ARMS = ['random', 'margin_tta_div', 'seq_margin', 'seq_pair', 'seq_pair_tta', 'seq_pair_div']


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
    ap.add_argument("--budgets", type=str, default="2,4,8")
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
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']

        # ---- frozen probe predictions on the pool ----
        Lp = Xp.float() @ W0c
        pred_p = Lp.argmax(1)
        top2p = torch.topk(Lp, 2, dim=1)
        margin_p = (top2p.values[:, 0] - top2p.values[:, 1]).abs()
        a_p = top2p.indices[:, 0]; b_p = top2p.indices[:, 1]

        # ---- candidate set: the boundary region (top cand_frac by low margin) ----
        n_cand = max(int(args.cand_frac * len(Xp)), 8 * bmax)
        cand = torch.argsort(margin_p)[:n_cand]
        cand_margin = margin_p[cand]
        Xcand = Xp[cand].float()

        # ---- label-free scores on the candidate set ----
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)

        cand_pred = pred_p[cand]
        cand_a = a_p[cand]; cand_b = b_p[cand]
        cand_pl = pl[cand]

        # ---- acquisition selectors (return CANDIDATE indices) ----
        def select_random(b):
            torch.manual_seed(7)
            return torch.randperm(len(cand))[:b]

        def select_margin_tta_div(b):
            score = -m + v
            topM = torch.argsort(score, descending=True)[:8 * b]
            # farthest-point over the topM candidate POSITIONS (returns positions
            # within topM, mapped back to candidate positions)
            feats = F.normalize(pool_f[cand[topM]].float(), p=2, dim=1).to(device)
            torch.manual_seed(3)
            s = [int(torch.randint(len(topM), (1,)).item())]
            dist = (feats - feats[s[0]]).norm(dim=1)
            for _ in range(b - 1):
                nxt = int(dist.argmax().item())
                s.append(nxt)
                d2 = (feats - feats[nxt]).norm(dim=1)
                dist = torch.minimum(dist, d2)
            return topM[torch.tensor(s)]

        def select_seq(b, mode):
            """Sequential acquisition over the candidate set. mode:
              'seq_margin'    always lowest margin (no pair focus)
              'seq_pair'      focus on a revealed error pair (a,b) -> next query
                              is the lowest-margin candidate with top-2 = (a,b)
              'seq_pair_tta'  same, but pick by margin + TTA instability within
                              the pair's boundary points
              'seq_pair_div'  same, but pick by farthest-point within the pair
            """
            torch.manual_seed(3)
            sel = []                     # candidate indices (positions in cand)
            in_sel = set()
            revealed = []                # revealed error pairs (a,b)
            # pair -> candidate positions with that top-2 pair
            pair_pos = {}
            for i in range(len(cand)):
                key = (int(cand_a[i]), int(cand_b[i]))
                pair_pos.setdefault(key, []).append(i)
            for _ in range(b):
                # pick the next candidate
                nxt = None
                if mode != 'seq_margin':
                    for (a, bb) in revealed:
                        members = [i for i in pair_pos.get((a, bb), []) if i not in in_sel]
                        if not members:
                            continue
                        if mode == 'seq_pair':
                            # lowest-margin member of the revealed pair
                            nxt = members[torch.argsort(cand_margin[members])[0].item()]
                        elif mode == 'seq_pair_tta':
                            # margin + TTA instability within the pair
                            score = -m[members] + v[members]
                            nxt = members[torch.argmax(score).item()]
                        elif mode == 'seq_pair_div':
                            # farthest-point from the already-selected labels
                            # within the pair's members
                            if len(in_sel) == 0:
                                nxt = members[0]
                            else:
                                cf = F.normalize(pool_f[cand[torch.tensor(members)]].float(), p=2, dim=1).to(device)
                                sel_feats = F.normalize(pool_f[cand[torch.tensor(sorted(in_sel))]].float(), p=2, dim=1).to(device)
                                dmin = (cf.unsqueeze(1) - sel_feats.unsqueeze(0)).norm(dim=2).min(dim=1).values
                                nxt = int(members[dmin.argmax().item()])
                        break
                if nxt is None:
                    # fall back: lowest-margin unselected candidate
                    order = torch.argsort(cand_margin)
                    for i in order.tolist():
                        if i not in in_sel:
                            nxt = i
                            break
                if nxt is None:
                    break
                sel.append(nxt); in_sel.add(nxt)
                # the label reveals the (pred,true) pair; if error, focus it
                if cand_pred[nxt] != cand_pl[nxt]:
                    revealed.append((int(cand_pred[nxt]), int(cand_pl[nxt])))
            return torch.tensor(sel)

        cond_res = {'refs': refs, 'gap': float(gap), 'arms': {}}
        for arm in arms:
            entry = {}
            for b in budgets:
                if arm == 'random':
                    sel_c = select_random(b)
                elif arm == 'margin_tta_div':
                    sel_c = select_margin_tta_div(b)
                elif arm == 'seq_margin':
                    sel_c = select_seq(b, 'seq_margin')
                elif arm == 'seq_pair':
                    sel_c = select_seq(b, 'seq_pair')
                elif arm == 'seq_pair_tta':
                    sel_c = select_seq(b, 'seq_pair_tta')
                elif arm == 'seq_pair_div':
                    sel_c = select_seq(b, 'seq_pair_div')
                else:
                    raise ValueError(arm)
                sel = cand[sel_c.long()]
                X_lab = Xp[sel].float().to(device); Y_lab = onehot(pl[sel], NUM_CLASSES).float().to(device)
                resid = (Y_lab - X_lab @ W0c.to(device))
                G = (X_lab @ U_or_g).t() @ resid
                gcs = {}
                for rho in rho_sweep:
                    Gn = G / (G.norm() + 1e-8)
                    W1 = W0c + (U_or_g @ (rho * Gn)).cpu().float()
                    d = mw(W1, Xv, vl) - refs['frozen']
                    gcs[str(rho)] = {'delta': float(d),
                                     'gap_closed': float(d / gap) if gap > 1e-9 else None}
                best = max((v['gap_closed'] or -9 for v in gcs.values()), default=None)
                # sequential diagnostics
                n_pair_focused = 0
                if arm.startswith('seq'):
                    # how many queries landed on a revealed pair (excluding the first)
                    if arm == 'seq_margin':
                        n_pair_focused = 0
                    else:
                        revealed = set()
                        for i in sel_c.tolist():
                            if (int(cand_pred[i]), int(cand_pl[i])) in revealed:
                                n_pair_focused += 1
                            if cand_pred[i] != cand_pl[i]:
                                revealed.add((int(cand_pred[i]), int(cand_pl[i])))
                # boundary share: fraction of labels on the top-2 boundary region
                bdry_share = float((margin_p[sel] <= torch.quantile(margin_p, 0.05)).float().mean().item())
                entry[str(b)] = {'gc': gcs, 'best_gc': best,
                                 'n_pair_focused': n_pair_focused,
                                 'bdry_share': bdry_share,
                                 'n_labels': int(len(sel))}
            cond_res['arms'][arm] = entry
            line = " ".join(f"b{b}:{v['best_gc']:+.2f}" for b, v in entry.items())
            print(f"    {arm:14s} {line}")

        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, U_or, Lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        for arm in arms:
            e = cond_res['arms'][arm]
            line = " ".join(f"b{b}:{v['best_gc']:+.2f}" for b, v in e.items())
            print(f"    {arm:14s} {line}")
        for arm in arms:
            e = cond_res['arms'][arm]
            foc = " ".join(f"b{b}:{v['n_pair_focused']}" for b, v in e.items())
            print(f"    {arm:14s} pair_focus {foc}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Same downstream (oracle-U first-order) for every arm; only the ACQUISITION")
    print("varies. The decisive number is best_gc per budget:")
    print("  seq_* > margin_tta_div  -> sequential labels reveal where to look next")
    print("  seq_* ~ margin_tta_div  -> the adaptive loop adds nothing over a good")
    print("                              one-shot rule")
    print("  seq_* < random          -> the loop actively hurts")
    print("  n_pair_focused          -> how often a label revealed an error pair and")
    print("                              the next query followed it (the mechanism)")


if __name__ == "__main__":
    main()
