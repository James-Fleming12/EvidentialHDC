"""probe_update_efficiency_diag.py: accuracy x efficiency table for the ridge
update, benchmarked against the R1 prototype pipeline (eval-only).

The gradient-free update W = (X^T X + lI)^{-1} X^T Y (primal) can be rewritten in
two algebraically IDENTICAL forms that change the cost shape:
  - DUAL  (Woodbury):  W = X^T (X X^T + lI_n)^{-1} Y. Flips the inversion from the
    feature dimension d=10000 (O(d^3)) to the SAMPLE dimension n (O(n^3)), so it is
    the right form when n << d (small pools / chunked updates).
  - RLS   (Sherman-Morrison): maintain P = (S + lI)^{-1} incrementally,
    P <- P - P h h^T P / (1 + h^T P h),  W <- W + P h (y - W^T h)^T.
    O(d^2) per point, NO solve ever -- the right form for point-by-point streaming.

All three give the SAME W (to numerical precision) for the same pool, so accuracy is
ONE curve vs pool size and the choice is purely efficiency.

For each condition and pool size, this reports:
  - mIoU (must match across primal/dual/RLS at the same n)
  - wall-clock of the update (accumulate + solve)
  - THROUGHPUT (points/s) of the update: pool_size / wall_s
  - peak RSS
plus the R1 PROTOTYPE reference (the existing pipeline): build_hdc_prototypes
(fit), weighted_mean_update (the label-free update), and the cosine decode, each
with wall-clock and points/s -- so the probe's overhead vs the prototype update is
directly comparable.

Usage:
  uv run python robust_diagnostic/probe_update_efficiency_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_update_efficiency_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import resource
import yaml
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 weighted_mean_update, compute_miou)

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


def predict_ridge(codes, W):
    W = W.detach().cpu()
    scores = codes.float() @ W
    return scores.argmax(dim=1)


def proto_lbls_argmax(sims):
    return sims.argmax(dim=1).cpu()


# ---------------- the three probe update forms ----------------

def primal(X, Y, lam, device):
    """W = (X^T X + lI)^{-1} X^T Y. Accumulate S/T in chunks, one d x d solve."""
    d = X.shape[1]
    S = torch.zeros(d, d, device=device)
    T = torch.zeros(d, Y.shape[1], device=device)
    t0 = time.time()
    for s in range(0, len(X), 50000):
        xc = X[s:s + 50000].float().to(device)
        S += xc.T @ xc
        T += xc.T @ Y[s:s + 50000].float().to(device)
    t_acc = time.time() - t0
    t0 = time.time()
    W = torch.linalg.solve(S + lam * torch.eye(d, device=device), T)
    t_solve = time.time() - t0
    return W, t_acc, t_solve


def dual(X, Y, lam, device):
    """W = X^T (X X^T + lI_n)^{-1} Y. Flip inversion to the sample dimension n."""
    t0 = time.time()
    Xf = X.float().to(device)
    G = Xf @ Xf.T                                  # n x n
    t_acc = time.time() - t0
    t0 = time.time()
    n = X.shape[0]
    A = torch.linalg.solve(G + lam * torch.eye(n, device=device), Y.float().to(device))
    W = Xf.T @ A
    t_solve = time.time() - t0
    return W, t_acc, t_solve


def rls(X, Y, lam, device):
    """RLS / Sherman-Morrison: maintain P = (S + lI)^{-1} and W incrementally,
    one point at a time, O(d^2) per point, no solve."""
    d = X.shape[1]
    C = Y.shape[1]
    P = torch.eye(d, device=device) / lam
    W = torch.zeros(d, C, device=device)
    t0 = time.time()
    for i in range(len(X)):
        h = X[i:i + 1].float().to(device).T          # (d, 1)
        Ph = P @ h
        denom = 1.0 + (h.T @ Ph).item()
        P = P - (Ph @ Ph.T) / denom
        err = (Y[i:i + 1].float().to(device).T - W.T @ h)
        W = W + (P @ h) @ err.T
    t_loop = time.time() - t0
    return W, 0.0, t_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--pool_sizes", type=str, default="1000,5000,10000,50000")
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_update_efficiency_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    pool_sizes = [int(x) for x in args.pool_sizes.split(',')]
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'd': fa.shape[1], 'conds': {}}

    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
        val_codes = hdc_codes(val, proj, device)

        pmax = pool_sizes[-1]
        pool, pl = f[perm[:pmax]], l[perm[:pmax]]
        pool_codes = hdc_codes(pool, proj, device)

        # LR oracle reference (the C10 ceiling) at the largest pool
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(pool_codes.numpy(), pl.numpy())
        lr_oracle = compute_miou(torch.tensor(lr.predict(val_codes.numpy())), vl)

        # ---- R1 prototype reference (the existing pipeline) ----
        ref = {}
        t0 = time.time()
        base_protos, base_lbls = build_hdc_prototypes(pool, pl, proj, device=device)
        ref['proto_fit_s'] = time.time() - t0
        ref['proto_fit_pts_s'] = len(pool) / ref['proto_fit_s']
        t0 = time.time()
        # the label-free prototype update = re-estimate with true labels (oracle form)
        w = torch.ones(len(pool))
        adapted = weighted_mean_update(base_protos, base_lbls, pool, pl.to(device), w, proj, device)
        ref['proto_update_s'] = time.time() - t0
        ref['proto_update_pts_s'] = len(pool) / ref['proto_update_s']
        t0 = time.time()
        preds = []
        for s in range(0, len(val_codes), 100000):
            hc = F.normalize(val_codes[s:s + 100000].to(device), p=2, dim=1)
            preds.append(base_lbls[(hc @ F.normalize(base_protos, p=2, dim=1).T).argmax(dim=1)].cpu())
        preds = torch.cat(preds)
        ref['proto_decode_s'] = time.time() - t0
        ref['proto_decode_pts_s'] = len(val_codes) / ref['proto_decode_s']
        ref['proto_miou'] = compute_miou(preds, vl)

        rows = []
        for n in pool_sizes:
            if n > len(pool_codes):
                continue
            ci = torch.randperm(len(pool_codes))[:n]
            X = pool_codes[ci]
            Y = onehot(pl[ci], NUM_CLASSES)
            row = {'pool': n}
            W_primal = W_dual = W_rls = None
            # primal
            try:
                W, ta, ts = primal(X, Y, args.lam, device)
                wall = ta + ts
                row['primal'] = {'miou': compute_miou(predict_ridge(val_codes, W), vl),
                                 'accum_s': ta, 'solve_s': ts, 'wall_s': wall,
                                 'pts_s': n / wall if wall > 0 else None}
                W_primal = W
            except Exception as e:
                row['primal'] = {'error': str(e)}
            # dual
            try:
                Wd, tda, tds = dual(X, Y, args.lam, device)
                wall = tda + tds
                row['dual'] = {'miou': compute_miou(predict_ridge(val_codes, Wd), vl),
                               'accum_s': tda, 'solve_s': tds, 'wall_s': wall,
                               'pts_s': n / wall if wall > 0 else None}
                W_dual = Wd
            except Exception as e:
                row['dual'] = {'error': str(e)}
            # RLS (small n only; sequential O(d^2) is slow)
            if n <= 2000:
                try:
                    Wr, tra, trs = rls(X, Y, args.lam, device)
                    wall = tra + trs
                    row['rls'] = {'miou': compute_miou(predict_ridge(val_codes, Wr), vl),
                                  'accum_s': tra, 'solve_s': trs, 'wall_s': wall,
                                  'pts_s': n / wall if wall > 0 else None}
                    W_rls = Wr
                except Exception as e:
                    row['rls'] = {'error': str(e)}
            else:
                row['rls'] = {'skipped': 'n>2000'}

            if W_primal is not None and W_dual is not None:
                row['max|W_primal-W_dual|'] = float((W_primal.cpu() - W_dual.cpu()).abs().max().item())
            if W_primal is not None and W_rls is not None:
                row['max|W_primal-W_rls|'] = float((W_primal.cpu() - W_rls.cpu()).abs().max().item())
            rows.append(row)

        results['conds'][cond] = {'lr_oracle_%d' % pmax: lr_oracle, 'proto_ref': ref, 'rows': rows}

        print(f"\n{'='*112}")
        print(f"=== {args.label} [{cond}]  d={fa.shape[1]}  LR-oracle(pool {pmax})={lr_oracle:.4f} ===")
        print(f"{'='*112}")
        print(f"  R1 prototype ref: fit {ref['proto_fit_s']:.3f}s ({ref['proto_fit_pts_s']:,.0f} pts/s) | "
              f"update {ref['proto_update_s']:.4f}s ({ref['proto_update_pts_s']:,.0f} pts/s) | "
              f"decode {ref['proto_decode_s']:.3f}s ({ref['proto_decode_pts_s']:,.0f} pts/s) | "
              f"mIoU {ref['proto_miou']:.4f}")
        print(f"  {'pool':>6} | {'primal mIoU':>11} {'pts/s':>12} | {'dual mIoU':>9} {'pts/s':>12} | {'RLS mIoU':>8} {'pts/s':>12}")
        for row in rows:
            p = row.get('primal', {}); d = row.get('dual', {}); r = row.get('rls', {})
            pv = f"{p.get('miou', float('nan')):.4f}" if 'miou' in p else p.get('error', 'skip')
            ps = f"{p.get('pts_s', 0):,.0f}" if 'pts_s' in p else '-'
            dv = f"{d.get('miou', float('nan')):.4f}" if 'miou' in d else d.get('error', 'skip')
            ds = f"{d.get('pts_s', 0):,.0f}" if 'pts_s' in d else '-'
            rv = f"{r.get('miou', float('nan')):.4f}" if 'miou' in r else r.get('skipped', r.get('error', 'skip'))
            rs = f"{r.get('pts_s', 0):,.0f}" if 'pts_s' in r else '-'
            print(f"  {row['pool']:>6} | {pv:>11} {ps:>12} | {dv:>9} {ds:>12} | {rv:>8} {rs:>12}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ THE TABLE ===")
    print("mIoU matches across primal/dual/RLS at the same pool (same W). Throughput (pts/s)")
    print("is the efficiency axis, compared to the R1 prototype update:")
    print("  - proto_update pts/s is the existing pipeline's update throughput.")
    print("  - the probe forms' pts/s show the overhead of the linear update.")
    print("  - dual wins at small n (n^3 vs d^3 inversion); RLS is streaming; primal baseline.")


if __name__ == "__main__":
    main()
