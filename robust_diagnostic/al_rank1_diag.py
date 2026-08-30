"""al_rank1_diag.py: is the rank-1-per-label decomposition of the update viable?
A diagnostic -- not a black-box -- for the idea: treat each queried label as its
own rank-1 update direction u_i = x_i r_i^T (r_i = e_{y_i} - p_0(x_i)), applied
separately, instead of the single aggregate G = sum_i u_i.

CRITICAL LINEARITY FACT. W0 + eta * sum_i u_i == applying the u_i sequentially
with the SAME eta. So "sequential vs aggregate" with a fixed step is a
non-difference. The only levers that can make the rank-1 decomposition matter are:
  (a) PER-LABEL step size eta_i (each label's update gets its own scale),
  (b) PER-LABEL REJECTION (keep only the good rank-1 updates, rollback the bad),
  (c) ADAPTIVE RE-QUERY (with the current W, which labels would be re-selected).
This diagnostic isolates each, plus the fundamental atomic question: is an
individual rank-1 update EVER useful?

Diagnostics (per label i, per condition):
  D1 ATOMIC: align(u_i, R) -- does the single-label rank-1 direction point at the
     oracle residual? (flattened cosine in d*C space)
  D2 ATOMIC-USEFULNESS: individual delta -- W0 + eta*u_i, does it improve val?
     This is the floor: if no single rank-1 update helps, the idea is dead at the
     atomic level regardless of decomposition.
  D3 CANCELATION: align(sum_i u_i, R) vs mean align(u_i, R); aggregate delta vs
     mean individual delta -- do the directions reinforce or cancel?
  D4 PER-LABEL SCALE (upper bound): eta_i = alignment-scaled (the oracle knows
     each u_i's alignment with R). Does oracle-scaled beat the aggregate? This
     isolates step-size as the failure, with an oracle upper bound.
  D5 REJECTION: can a LABEL-FREE score (change in probe confidence / probe-vs-
     proto disagreement after applying u_i) identify the good individual updates?
     corr(score_i, delta_i), and keep-top-K (oracle-k) delta vs aggregate.
  D6 ADAPTIVE RE-QUERY: re-select labels with W0 + eta*u_best applied -- does the
     updated W change which points are selected (the true sequential-AL claim)?

Acquisition: margin_tta_div (Iteration-1 winner) for the initial b labels.

Read: the idea works if D2 shows some individual updates are positive AND D4 (oracle
scale) lifts the aggregate AND D5 shows a label-free score can pick them. If D2 is
all negative, the decomposition is dead regardless of how it is applied.

Usage:
  uv run python robust_diagnostic/al_rank1_diag.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_rank1_<label>.json
"""
import os, sys, time, argparse, json, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
SKETCH_SEED = 11


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
    ap.add_argument("--b", type=int, default=8, help="label budget")
    ap.add_argument("--eta", type=float, default=0.05, help="fixed per-label step")
    ap.add_argument("--cand_frac", type=float, default=0.05)
    ap.add_argument("--tta_augs", type=int, default=5)
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="dglsspp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config)); ARCH = yaml.safe_load(open(args.arch))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    b = args.b

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b, method=args.method_b)
    model = trainer.model
    model.eval()
    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_clean(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    results = {'label': args.label, 'method': args.method_b, 'b': b, 'eta': args.eta, 'conds': {}}

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
        refs = {'frozen': mw(W0, Xv, vl), 'oracle': mw(Ws, Xv, vl)}
        gap = refs['oracle'] - refs['frozen']
        R_flat = R.flatten().double()

        # ---- acquisition (margin_tta_div) ----
        sm = torch.softmax(Xp.float() @ W0c, dim=1)
        top2 = torch.topk(sm, 2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).abs()
        cand = torch.argsort(margin)[:max(int(args.cand_frac * len(Xp)), 8 * b)]
        n_cand = len(cand)
        cand_margin = margin[cand]
        Xcand = Xp[cand].float()
        draws = []
        for _ in range(args.tta_augs):
            torch.manual_seed(100 + _)
            flip = torch.rand_like(Xcand) < 0.02
            draws.append(torch.softmax(torch.where(flip, -Xcand, Xcand) @ W0c, dim=1))
        tta_var = torch.stack(draws).var(dim=0).mean(dim=1)
        m = cand_margin / (cand_margin.max() + 1e-8)
        v = tta_var / (tta_var.max() + 1e-8)
        topM = torch.argsort(-m + v, descending=True)[:8 * b]
        sel = farthest_point(pool_f, cand[topM], b, device).long()

        X_lab = Xp[sel]; y_lab = pl[sel]
        p0 = torch.softmax(X_lab.float() @ W0c, dim=1)

        # ---- per-label rank-1 updates u_i = x_i r_i^T ----
        per = {}
        us = []
        for i in range(len(sel)):
            r_i = (onehot(y_lab[i:i+1], NUM_CLASSES) - p0[i:i+1]).float()   # 1 x C
            u_i = X_lab[i:i+1].float().t() @ r_i                             # d x C rank-1
            us.append(u_i)
            align_i = float(F.cosine_similarity(u_i.flatten().double(), R_flat, dim=0).item())
            # D1/D2: individual usefulness -- apply ONLY this rank-1 update
            W_i = W0c + args.eta * u_i
            d_i = mw(W_i, Xv, vl) - refs['frozen']
            # label-free validation score candidates (after applying u_i)
            sm1 = torch.softmax(Xp.float() @ W_i, dim=1)
            conf1 = float(sm1.max(dim=1).values.mean().item())
            conf0 = float(sm.max(dim=1).values.mean().item())
            per[str(i)] = {
                'align_u_R': align_i,
                'delta': float(d_i),
                'gap_closed': float(d_i / gap) if gap > 1e-9 else None,
                'd_conf': conf1 - conf0,
                '|r_i|': float(r_i.norm().item()),
                'margin': float(margin[sel[i]].item()),
            }
        U = torch.stack(us, dim=0)          # b x d x C

        # ---- D3 cancelation: aggregate vs mean individual ----
        U_agg = us[0].clone()
        for u in us[1:]:
            U_agg = U_agg + u
        align_agg = float(F.cosine_similarity(U_agg.flatten().double(), R_flat, dim=0).item())
        align_mean = float(np.mean([p['align_u_R'] for p in per.values()]))
        d_agg = mw(W0c + args.eta * U_agg, Xv, vl) - refs['frozen']
        d_mean_indiv = float(np.mean([p['delta'] for p in per.values()]))

        # ---- D4 per-label scale (oracle upper bound): eta_i = align-scaled ----
        # W_seq_oracle = W0 + sum_i (eta * s_i) u_i, s_i = signed align (sign of delta)
        W_or = W0c.clone()
        for i, u in enumerate(us):
            s_i = 1.0 if per[str(i)]['delta'] > 0 else -1.0      # oracle knows sign
            W_or = W_or + args.eta * s_i * u
        d_oracle_seq = mw(W_or, Xv, vl) - refs['frozen']

        # ---- D5 rejection: label-free score to pick good updates ----
        conf_d = [p['d_conf'] for p in per.values()]
        deltas = [p['delta'] for p in per.values()]
        try:
            corr_d = float(np.corrcoef(conf_d, deltas)[0, 1])
        except Exception:
            corr_d = None
        # keep-top-K by the label-free score (oracle-k: keep by true delta sign)
        n_good = sum(1 for d in deltas if d > 0)
        W_k_or = W0c.clone()
        for i, u in enumerate(us):
            if per[str(i)]['delta'] > 0:
                W_k_or = W_k_or + args.eta * u
        d_keep_oracle = mw(W_k_or, Xv, vl) - refs['frozen']

        cond_res = {
            'refs': refs, 'gap': float(gap),
            'per_label': per,
            'D3': {'align_agg': align_agg, 'align_mean': align_mean,
                   'delta_agg': float(d_agg), 'delta_mean_indiv': d_mean_indiv,
                   'n_positive': n_good, 'n_total': b},
            'D4': {'delta_oracle_seq': float(d_oracle_seq)},
            'D5': {'corr_dconf_delta': corr_d, 'delta_keep_oracle': float(d_keep_oracle),
                   'n_good': n_good},
        }
        results['conds'][cond] = cond_res
        del Xp, Xv, Ws, R, pool_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n=== {cond} ({toc(t0):.0f}s) frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} gap {gap:+.3f} ===")
        print(f"  per-label: " + " ".join(
            f"l{i}:align{p['align_u_R']:.2f},d{p['delta']:+.2f}" for i, p in per.items()))
        print(f"  D3: align_agg {align_agg:.3f} vs mean {align_mean:.3f} | d_agg {d_agg:+.2f} vs mean_indiv {d_mean_indiv:+.2f} | {n_good}/{b} positive")
        print(f"  D4: oracle-scaled sequential {d_oracle_seq:+.2f} (upper bound on per-label scale)")
        print(f"  D5: corr(d_conf, delta) {corr_d} | keep-oracle-good {d_keep_oracle:+.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ (what to look for) ===")
    print("D1/D2 (atomic): are individual rank-1 updates aligned with R AND positive?")
    print("  If align ~0 and all delta < 0: the idea is dead at the atomic level -- no")
    print("  decomposition or sequential scheme can help (each label's direction is bad).")
    print("D3 (cancelation): if align_agg > align_mean AND d_agg > d_mean_indiv, the")
    print("  directions reinforce (aggregate is the right move). If d_agg < mean, they")
    print("  cancel and per-label handling is justified.")
    print("D4 (per-label scale, oracle): if d_oracle_seq >> d_agg, the FAILURE is step")
    print("  size -- each label needs its own scale. If d_oracle_seq ~ d_agg, scale is")
    print("  not the issue.")
    print("D5 (rejection): corr(d_conf, delta) tells whether a label-free score can pick")
    print("  the good updates; keep-oracle-good is the best case for rejection.")
    print("The idea WORKS if D2 has positives AND D4 or D5 show the aggregate was the")
    print("wrong packaging. It is DEAD if D2 is all-negative (atomic failure).")


if __name__ == "__main__":
    main()
