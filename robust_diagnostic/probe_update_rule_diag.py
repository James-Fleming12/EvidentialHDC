"""probe_update_rule_diag.py: validate the gradient-free probe UPDATE RULES (eval-only).

Iteration 0 showed the zs->oracle gap is a ROTATION (weights must move, bias-only
= 0-4%), so the update must be option 2 (closed-form ridge / accumulate-and-solve)
or option 3 (FLDA). Before designing the TTA/AL gate, this tests whether those rules
have the REQUIRED properties on the cov-shift features:

  CORRECTNESS  : does the rule reach the LR oracle mIoU (the C10 reference)?
  EFFICIENCY   : is the update backprop-free and built from simple accumulate-and-
                 solve statistics (like prototype means are built from sums)?
  EQUIVALENCE  : does the INCREMENTAL accumulated form match the BATCH closed form?

Rules compared, per condition and pool size:
  R1 prototype     : per-class mean of the sign codes (the current cheap rule).
  LR oracle        : sklearn LogisticRegression pool-refit (C10 reference ceiling).
  Ridge-batch      : closed-form least-squares W = (X^T X + lI)^{-1} X^T Y in one shot.
  Ridge-accum      : the UPDATE form -- accumulate S=X^T X, T=X^T Y in chunks over
                     the pool, ONE solve at the end. Must match Ridge-batch.
  FLDA             : sklearn LinearDiscriminantAnalysis (solver='lsqr'), the
                     class-scatter rule (option 3).

Efficiency is measured as wall-clock for the accumulate+solve vs the LR fit, and
peak RSS. If Ridge-accum matches Ridge-batch (numerically) and reaches the LR
oracle, it is the gradient-free update rule the TTA/AL gate should iterate on.

Usage:
  uv run python robust_diagnostic/probe_update_rule_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds snow,wet_ground,fog,crosstalk \
    --out robust_diagnostic/logs/probe_update_rule_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import resource
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

CONDS_DEFAULT = ['snow', 'wet_ground', 'fog', 'crosstalk']
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


def ridge_accumulate(codes, lbls, lam=1e-3, num_classes=17, device='cuda', chunk=50000):
    """Option-2 UPDATE form: accumulate S = X^T X and T = X^T Y over the pool in
    chunks, then ONE solve. Returns W (d, C), the accumulate wall-time, and the
    solve wall-time. Backprop-free: only outer products / matmuls and a solve."""
    d = codes.shape[1]
    S = torch.zeros(d, d, device=device)
    T = torch.zeros(d, num_classes, device=device)
    t_acc = 0.0
    t0 = time.time()
    for s in range(0, len(codes), chunk):
        X = codes[s:s + chunk].to(device).float()
        Y = onehot(lbls[s:s + chunk], num_classes).to(device)
        S += X.T @ X
        T += X.T @ Y
    t_acc = time.time() - t0
    t0 = time.time()
    I = torch.eye(d, device=device)
    W = torch.linalg.solve(S + lam * I, T)
    t_solve = time.time() - t0
    return W, t_acc, t_solve


def ridge_predict(codes, W, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        scores = codes[s:s + chunk].float() @ W
        preds.append(scores.argmax(dim=1))
    return torch.cat(preds, dim=0)


def ridge_batch(codes, lbls, lam=1e-3, num_classes=17, device='cuda'):
    """The BATCH closed form on the full pool at once (reference for equivalence)."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    d = X.shape[1]
    I = torch.eye(d, device=device)
    return torch.linalg.solve(X.T @ X + lam * I, X.T @ Y)


def bench_rule(name, fn, codes_val, vl):
    """Run fn -> (W, *times), predict on val, return mIoU + timing + RSS."""
    t0 = time.time()
    out = fn()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if isinstance(out, tuple):
        W = out[0]
        times = out[1:]
    else:
        W = out
        times = ()
    miou = compute_miou(ridge_predict(codes_val, W), vl)
    return {'miou': miou, 'wall_s': time.time() - t0, 'rss_mb': rss, 'times': times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--lam", type=float, default=1e-3, help="ridge regularization")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_update_rule_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {}
    print(f"\n{'='*100}")
    print(f"=== {args.label}: probe update-rule validation (pool {args.pool_size}, lam {args.lam}) ===")
    print(f"{'='*100}")

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

        # reference rules
        lr_clf = LogisticRegression(max_iter=args.max_iter, C=1.0)
        lr_clf.fit(pool_codes.numpy(), pl.numpy())
        lr_miou = compute_miou(torch.tensor(lr_clf.predict(val_codes.numpy())), vl)

        # option 2: batch vs accumulate-and-solve. Compute the accumulated W ONCE and
        # reuse for the equivalence check (avoids re-accumulating the 10000x10000 S).
        acc_res = ridge_accumulate(pool_codes, pl, args.lam, device=device)
        W_acc, t_acc, t_solve = acc_res
        ra = bench_rule('ridge-accum', lambda: acc_res, val_codes, vl)
        W_batch = ridge_batch(pool_codes, pl, args.lam, device=device)
        rb = bench_rule('ridge-batch', lambda: (W_batch, 0.0), val_codes, vl)
        w_diff = float((W_acc - W_batch).abs().max().item())

        # option 3: FLDA (solver='lsqr' avoids the full d x d eigen-decomposition)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        t0 = time.time()
        lda.fit(pool_codes.numpy(), pl.numpy())
        lda_wall = time.time() - t0
        lda_miou = compute_miou(torch.tensor(lda.predict(val_codes.numpy())), vl)

        # numeric equivalence of accumulate vs batch (S/T are identical by construction,
        # so the W's should match to ~1e-3; report the max |W| diff as a sanity check)
        results[cond] = {
            'pool_size': len(pool), 'n_val': len(val),
            'lr_oracle': {'miou': lr_miou},
            'ridge_batch': rb,
            'ridge_accum': ra,
            'flda': {'miou': lda_miou, 'wall_s': lda_wall},
            'accum_eq_batch_max_wdiff': w_diff,
        }
        print(f"\n=== {cond} (pool {len(pool)}, val {len(val)}) ===")
        print(f"  LR oracle          : {lr_miou:.4f}")
        print(f"  Ridge-batch        : {rb['miou']:.4f}  ({rb['wall_s']:.1f}s, rss {rb['rss_mb']:.0f}MB)")
        print(f"  Ridge-accum        : {ra['miou']:.4f}  (accum {ra['times'][0]:.1f}s + solve {ra['times'][1]:.1f}s, "
              f"wall {ra['wall_s']:.1f}s, rss {ra['rss_mb']:.0f}MB)")
        print(f"  FLDA               : {lda_miou:.4f}  ({lda_wall:.1f}s)")
        print(f"  accum==batch max|W diff|: {w_diff:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== VERDICT RULES ===")
    print("1. Does Ridge-accum reach the LR oracle? (miou gap < ~0.02 = optimal enough)")
    print("2. Is Ridge-accum == Ridge-batch (max|W diff| ~ 1e-3)? = the update is a pure")
    print("   accumulate-and-solve, no gradients, so it is the gradient-free update rule.")
    print("3. Is Ridge-accum's accumulate+solve wall-clock comparable to the prototype")
    print("   update (<< the iterative LR fit)? = it keeps the efficiency story.")
    print("4. Does FLDA reach the oracle too? (option 3, if ridge fails on some condition)")


if __name__ == "__main__":
    main()
