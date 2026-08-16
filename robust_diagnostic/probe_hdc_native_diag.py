"""probe_hdc_native_diag.py: HDC-native efficiency for the linear probe (eval-only).

Iteration 2 showed the 7-8x overhead of the 10000-d probe can be removed by
shrinking the code (an implementation trick), but the paper's preferred direction
is to keep the 10000-d HDC space and make the LR calculations simpler / have a
simpler form USING THE BINARY SPACE. This scan measures those binary-space levers.

CLASSIFICATION (decode) in the binary space:
  - float W        : the ridge solution, decoded as float dot products (baseline).
  - sign W         : W quantized to +-1. Then W_c . h for +-1 code h is an INTEGER
                     in [-d, d] = d - 2*Hamming(W_c, h) -- a popcount on packed
                     bits. Does quantization keep the mIoU? (the HDC-native decode)

UPDATES in the binary space:
  - dual float     : W = X^T (X X^T + lI)^{-1} Y with a float n x n solve (baseline).
  - dual int G     : X X^T is a matrix of +-1 dot products = INTEGER entries; compute
                     G as integer matmul (exact, no float rounding) then solve.
  - block ridge    : split the 10000-d code into B blocks of d/B dims, fit an
                     independent ridge probe per block (each solve is (d/B)^3), and
                     decode as the sum of per-block scores. Turns one d^3 solve into
                     B * (d/B)^3 = d^3/B^2 -- the big win. Accuracy may drop if the
                     cross-block covariance (the probe's rotation) is block-diagonal
                     enough to be captured.
  - dual-RLS       : streaming in the SAMPLE space: maintain the n x n Gram inverse
                     incrementally (Sherman-Morrison on (X X^T + lI)^{-1}), O(n^2)
                     per point, no d^2 at all -- the sample-space streaming form.

For each: mIoU on val, update wall-clock, update pts/s, peak RSS, at a fixed pool.
Reported alongside the R1 prototype reference.

Usage:
  uv run python robust_diagnostic/probe_hdc_native_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_hdc_native_covshift_ep10.json
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


def mIoU_preds(preds, vl):
    return compute_miou(preds, vl)


# ---------------- decode variants ----------------

def decode_float(codes, W):
    W = W.detach().cpu()
    return (codes.float() @ W).argmax(dim=1)


def decode_sign(codes, W):
    """Quantize W to +-1, decode as integer dot products (d - 2*Hamming on packed
    bits in a real system). Scores are integers; argmax unchanged by the d offset."""
    Wq = W.detach().cpu().sign()
    return (codes.float() @ Wq).argmax(dim=1)


# ---------------- update variants (all return W, (times)) ----------------

def dual_float(codes, lbls, lam, device):
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    t0 = time.time()
    G = X @ X.T
    t_acc = time.time() - t0
    t0 = time.time()
    n = X.shape[0]
    A = torch.linalg.solve(G + lam * torch.eye(n, device=device), Y)
    W = X.T @ A
    return W, t_acc, time.time() - t0


def dual_int(codes, lbls, lam, device):
    """G = X X^T as INTEGER matmul (exact +-1 dot products), then the same n x n solve.
    Uses int32 so torch.matmul supports it; still exact integer (no float rounding)."""
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    t0 = time.time()
    Xi = codes.to(device).to(torch.int32)         # +-1 in int32
    G = (Xi @ Xi.T).float()                        # exact integer G, exact sum
    t_acc = time.time() - t0
    t0 = time.time()
    n = X.shape[0]
    A = torch.linalg.solve(G + lam * torch.eye(n, device=device), Y)
    W = X.T @ A
    return W, t_acc, time.time() - t0


def block_ridge(codes, lbls, lam, device, n_blocks=20):
    """Split d into n_blocks, fit an independent ridge per block, decode as the sum
    of per-block scores. Each solve is (d/n_blocks)^3."""
    d = codes.shape[1]
    bs = d // n_blocks
    W = torch.zeros(d, NUM_CLASSES, device=device)
    t_acc = 0.0
    t_solve = 0.0
    for b in range(n_blocks):
        sl = slice(b * bs, (b + 1) * bs) if b < n_blocks - 1 else slice(b * bs, d)
        Xb = codes[:, sl].float().to(device)
        Y = onehot(lbls, NUM_CLASSES).to(device)
        t0 = time.time()
        S = Xb.T @ Xb
        T = Xb.T @ Y
        t_acc += time.time() - t0
        t0 = time.time()
        db = S.shape[0]
        Wb = torch.linalg.solve(S + lam * torch.eye(db, device=device), T)
        W[sl] = Wb
        t_solve += time.time() - t0
    return W, t_acc, t_solve


def dual_rls(codes, lbls, lam, device):
    """Streaming in the SAMPLE space: append points one at a time, maintaining the
    n x n Gram inverse (X X^T + lI)^{-1} via the block-inverse update. Each new
    point is an O(n^2) update (rank-1 Sherman-Morrison on the Gram), never touching
    the d^2 / d^3 feature-space solve. The sample-space RLS.

    For a pool of n this is O(n^3) total (n updates of O(n^2)) -- the same as one
    n x n solve -- but it is INCREMENTAL: for streaming (few points between solves)
    it is the right cost shape."""
    n = codes.shape[0]
    X = codes.float()
    Y = onehot(lbls, NUM_CLASSES).float()
    Xm = X[0:1]
    Ginv = torch.eye(1, device=device) / (lam + (Xm @ Xm.T).item())
    t0 = time.time()
    for i in range(1, n):
        x = X[i:i + 1].to(device)                    # (1, d)
        g = (Xm @ x.T).to(device)                    # (i,1) old-vs-new Gram terms
        b = Ginv @ g                                 # (i,1)
        c = (x @ x.T).item() + lam
        S = c - (g.T @ b).item()                     # Schur complement
        top_left = Ginv + (b @ b.T) / S
        top_right = -(b / S)
        bot_left = -(b.T / S)
        bot_right = torch.ones(1, 1, device=device) / S
        Ginv = torch.cat([torch.cat([top_left, top_right], dim=1),
                          torch.cat([bot_left, bot_right], dim=1)], dim=0)
        Xm = torch.cat([Xm, x], dim=0)
    A = Ginv @ Y.to(device)
    W = Xm.T @ A
    return W, 0.0, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--n_blocks", type=int, default=20)
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_hdc_native_results.json")
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

    results = {'label': args.label, 'pool_size': args.pool_size,
               'n_blocks': args.n_blocks, 'conds': {}}

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

        # R1 prototype reference
        t0 = time.time()
        base_protos, base_lbls = build_hdc_prototypes(pool, pl, proj, device=device)
        proto_fit = time.time() - t0
        preds = []
        for s in range(0, len(val_codes), 100000):
            hc = F.normalize(val_codes[s:s + 100000].to(device), p=2, dim=1)
            preds.append(base_lbls[(hc @ F.normalize(base_protos, p=2, dim=1).T).argmax(dim=1)].cpu())
        proto_miou = compute_miou(torch.cat(preds), vl)

        rows = {}
        # --- decode variants ---
        for name, dec in [('float', decode_float), ('sign', decode_sign)]:
            for uname in ['dual_float', 'block_ridge']:
                if uname == 'dual_float':
                    W, ta, ts = dual_float(pool_codes, pl, args.lam, device)
                else:
                    W, ta, ts = block_ridge(pool_codes, pl, args.lam, device, args.n_blocks)
                wall = ta + ts
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                miou = mIoU_preds(dec(val_codes, W), vl)
                rows['decode_%s_%s' % (name, uname)] = {
                    'miou': miou, 'update_s': wall,
                    'update_pts_s': args.pool_size / wall if wall > 0 else None,
                    'rss_mb': rss}
        # --- update variants (float W decode for fair update comparison) ---
        for uname, fn in [('dual_float', dual_float), ('dual_int', dual_int),
                          ('block_ridge', lambda c, l, lam, d: block_ridge(c, l, lam, d, args.n_blocks))]:
            try:
                W, ta, ts = fn(pool_codes, pl, args.lam, device)
                wall = ta + ts
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                rows[uname] = {'miou': mIoU_preds(decode_float(val_codes, W), vl),
                               'update_s': wall,
                               'update_pts_s': args.pool_size / wall if wall > 0 else None,
                               'rss_mb': rss}
            except Exception as e:
                rows[uname] = {'error': str(e)}
        # dual-RLS streaming (small pool only; O(n^2) per point)
        if args.pool_size <= 2000:
            try:
                W, ta, ts = dual_rls(pool_codes, pl, args.lam, device)
                wall = ta + ts
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                rows['dual_rls'] = {'miou': mIoU_preds(decode_float(val_codes, W), vl),
                                    'update_s': wall,
                                    'update_pts_s': args.pool_size / wall if wall > 0 else None,
                                    'rss_mb': rss}
            except Exception as e:
                rows['dual_rls'] = {'error': str(e)}
        else:
            rows['dual_rls'] = {'skipped': 'n>2000'}

        results['conds'][cond] = {'proto': {'miou': proto_miou, 'fit_s': proto_fit},
                                  'rows': rows}

        print(f"\n{'='*110}")
        print(f"=== {args.label} [{cond}]  pool {args.pool_size} ===")
        print(f"{'='*110}")
        print(f"  R1 prototype: mIoU {proto_miou:.4f}  fit {proto_fit:.3f}s")
        print(f"  {'method':<28} {'mIoU':>7} {'update_s':>9} {'pts/s':>11}")
        for name, r in rows.items():
            if 'miou' in r:
                print(f"  {name:<28} {r['miou']:>7.4f} {r['update_s']:>9.3f} {r['update_pts_s']:>11,.0f}")
            else:
                print(f"  {name:<28} {r.get('error', r.get('skipped', 'skip'))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ THE SCAN ===")
    print("CLASSIFICATION (decode):")
    print("  decode_sign_* vs decode_float_*: if quantizing W to +-1 keeps mIoU, the")
    print("  decode is an integer dot (d - 2*Hamming, popcount on packed bits) -- the")
    print("  HDC-native decode, no floats.")
    print("UPDATES:")
    print("  dual_int vs dual_float: G = X X^T is an EXACT integer (+-1 dot) matrix;")
    print("  if equal mIoU, the update uses integer math (no float rounding).")
    print("  block_ridge: B small (d/B)^3 solves instead of one d^3 -- the big win;")
    print("  check mIoU (does the rotation survive block-diagonal approximation?).")
    print("  dual_rls: sample-space streaming, O(n^2)/point, no d^2 at all.")


if __name__ == "__main__":
    main()
