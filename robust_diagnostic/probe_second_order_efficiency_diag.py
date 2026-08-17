"""probe_second_order_efficiency_diag.py: cheaper second-order updates (eval-only).

Iteration 6 settled it: the probe's gain is cross-coordinate (second-order); no
first-order separator recovers it. So the question becomes making the SECOND-ORDER
problem cheaper, not finding another first-order form. This tests the Tier-1
directions:

  1. HARD-POINT CORESET + DUAL RIDGE (the feedback's "run immediately" pick).
     The oracle gain is concentrated on LOW-MARGIN points (Iteration 0: oracle-
     fixed points have margin ~6-8 vs ~12-16 for correct). The dual form
     W = X^T (X X^T + lI)^{-1} Y collapsed at n=10k only because of CONDITIONING
     (n too large), not the formulation. Select the m lowest-margin pool points
     (500/1k/2k/5k) and solve the m x m dual there -- the solve is tiny AND the
     classifier keeps the full d-dimensional cross-coordinate structure.
  2. MATRIX-FREE CG: solve (S + lI) W = T where Sv = X^T(Xv) -- never build the
     10k x 10k S, just one/two passes over the pool per CG iteration. With 5-20
     iterations this may approach the full ridge ceiling without d^2 storage.
  3. SPARSE COVARIANCE: approximate S by D + S_sparse (keep only the top-K
     off-diagonal entries by |S_jk|). Plot mIoU vs K -- does a small fraction of
     the cross-coordinate correlations carry the domain-specific structure?

Each reports mIoU (ceiling) + update wall-clock + pts/s, vs the R1 prototype and
full ridge references.

Usage:
  uv run python robust_diagnostic/probe_second_order_efficiency_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_second_order_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import torch

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'


def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)


def extract_features(model, parser, device, num_frames=100):
    feats, lbls = [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(parser.get_train_set()):
            if i >= num_frames:
                break
            in_vol = batch[0].to(device)
            labels = batch[2].to(device).view(-1)
            mask = (batch[1].to(device) > 0).view(-1)
            out_tuple = model(in_vol)
            if len(out_tuple) == 3:
                _, _, z8 = out_tuple
            else:
                _, z8 = out_tuple
            z_flat = z8.permute(0, 2, 3, 1).reshape(-1, z8.shape[1])[mask]
            feats.append(z_flat.cpu())
            lbls.append(labels[mask].cpu())
    return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)


def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)


def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y


def decode(W, codes, chunk=100000):
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)


def hard_point_indices(pool_codes, pl, mu, n_select):
    """Select the n_select LOWEST-margin pool points under the prototype decoder
    (confident-but-wrong boundary points -- where the oracle gain lives)."""
    mu_n = torch.nn.functional.normalize(mu, p=2, dim=1)
    scores = pool_codes.float() @ mu_n.T
    top2 = torch.topk(scores, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return torch.argsort(margin)[:n_select]


def dual_ridge(codes, lbls, lam, device, num_classes=NUM_CLASSES):
    """W = X^T (X X^T + lI)^{-1} Y on the (small) selected coreset. m x m solve."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    m = X.shape[0]
    t0 = time.time()
    G = X @ X.T
    t_acc = time.time() - t0
    t0 = time.time()
    A = torch.linalg.solve(G + lam * torch.eye(m, device=device), Y)
    W = X.T @ A
    t_solve = time.time() - t0
    return W, t_acc, t_solve


def matrix_free_cg(codes, lbls, lam, device, iters=10, num_classes=NUM_CLASSES):
    """CG solve (S + lI) W = T with Sv = X^T (X v) -- never build S. One forward +
    one backward pass over the pool per iteration. For +/-1 codes X v is a matmul."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    T = X.T @ Y
    d = X.shape[1]
    t0 = time.time()
    # explicit S for the RHS is only used to seed the check; matrix-free below
    S = X.T @ X
    b = T
    A = lambda v: S @ v                     # keep S for timing honesty; note: could be X^T(Xv)
    x = torch.zeros_like(b)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    t_solve = time.time() - t0
    return x, 0.0, t_solve, S.shape[0]


def sparse_cov_ridge(codes, lbls, lam, device, keep_frac, num_classes=NUM_CLASSES):
    """S ~ D + S_sparse: keep diagonal + the top-K off-diagonal |S_jk| entries.
    Solve (S_sparse + lI) W = T. keep_frac = fraction of off-diagonal entries kept."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    d = X.shape[1]
    t0 = time.time()
    S = X.T @ X
    T = X.T @ Y
    t_acc = time.time() - t0
    t0 = time.time()
    # mask off-diagonal to the top-K by magnitude
    S_mask = torch.zeros_like(S)
    S_mask.fill_diagonal_(1.0)
    if keep_frac > 0:
        off = S * (1 - S_mask)
        k = max(1, int(off.numel() * keep_frac))
        flat = off.abs().flatten()
        thresh = torch.topk(flat, k).values[-1]
        keep = off.abs() >= thresh
        S_mask = S_mask | keep
    S_sp = S * S_mask
    W = torch.linalg.solve(S_sp + lam * torch.eye(d, device=device), T)
    t_solve = time.time() - t0
    kept = int(S_mask.sum().item()) - d  # off-diagonal entries kept
    return W, t_acc, t_solve, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--coreset_sizes", type=str, default="500,1000,2000,5000")
    parser.add_argument("--cg_iters", type=str, default="5,10,20")
    parser.add_argument("--sparse_fracs", type=str, default="0.0,0.001,0.005,0.01,0.05")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_second_order_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    coreset_sizes = [int(x) for x in args.coreset_sizes.split(',')]
    cg_iters = [int(x) for x in args.cg_iters.split(',')]
    sparse_fracs = [float(x) for x in args.sparse_fracs.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        pool_codes = hdc_codes(pool, proj, device)
        val_codes = hdc_codes(val, proj, device)

        # prototype mean (for margin selection + the R1 baseline)
        mu = torch.zeros(NUM_CLASSES, pool_codes.shape[1])
        for c in range(1, NUM_CLASSES):
            m = pl == c
            if int(m.sum().item()) > 0:
                mu[c] = pool_codes[m].float().mean(dim=0)
        r1 = compute_miou(decode(mu.t(), val_codes), vl)

        # full ridge (ceiling)
        X = pool_codes.float()
        Y = onehot(pl, NUM_CLASSES).float()
        W_full = torch.linalg.solve(X.t() @ X + args.lam * torch.eye(10000), X.t() @ Y)
        full = compute_miou(decode(W_full, val_codes), vl)

        r = {'r1_proto': r1, 'full_ridge': full, 'coreset': {}, 'cg': {}, 'sparse': {}}

        # 1. hard-point coreset + dual ridge
        for m in coreset_sizes:
            idx = hard_point_indices(pool_codes, pl, mu, m)
            W, ta, ts = dual_ridge(pool_codes[idx], pl[idx], args.lam, device)
            r['coreset'][str(m)] = {
                'miou': compute_miou(decode(W, val_codes), vl),
                'solve_s': ta + ts, 'pts_s': m / (ta + ts) if (ta + ts) > 0 else None}

        # 2. matrix-free CG (using explicit S for a fair timing baseline; the code
        #    notes the X^T(Xv) alternative would skip d^2 storage)
        for it in cg_iters:
            W, ta, ts, _ = matrix_free_cg(pool_codes, pl, args.lam, device, it)
            r['cg'][str(it)] = {
                'miou': compute_miou(decode(W, val_codes), vl),
                'solve_s': ta + ts, 'pts_s': args.pool_size / (ta + ts) if (ta + ts) > 0 else None}

        # 3. sparse covariance
        for kf in sparse_fracs:
            W, ta, ts, kept = sparse_cov_ridge(pool_codes, pl, args.lam, device, kf)
            r['sparse'][str(kf)] = {
                'miou': compute_miou(decode(W, val_codes), vl),
                'solve_s': ta + ts, 'offdiag_kept': kept}

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  R1 proto {r1:.4f} | full ridge {full:.4f}")
        print(f"  coreset (hard-point + dual): " + "  ".join(
            f"m={m}:{r['coreset'][str(m)]['miou']:.4f}" for m in coreset_sizes))
        print(f"  CG iters: " + "  ".join(
            f"{it}:{r['cg'][str(it)]['miou']:.4f}" for it in cg_iters))
        print(f"  sparse cov (frac offdiag): " + "  ".join(
            f"{kf}:{r['sparse'][str(kf)]['miou']:.4f}" for kf in sparse_fracs))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("1. coreset: does a small low-margin coreset (m=500-2000) + dual solve reach")
    print("   near the full ridge ceiling? The solve is m x m (tiny) with full d-dims.")
    print("2. cg: does matrix-free CG (few iterations) approach the full ceiling? If it")
    print("   converges in 5-10 iters, it needs only ~10 passes over the pool, no d^2 S.")
    print("3. sparse: does a small fraction of off-diagonal S recover the gain? If 1%")
    print("   suffices, S is mostly diagonal + a few informative pairwise correlations.")


if __name__ == "__main__":
    main()
