"""probe_method_table_diag.py: README table for the HDC-native probe method (eval-only).

Produces, per condition, the zero-shot and ceiling (labeled oracle) mIoU and the
efficiency (update + decode throughput) for four decoders on the cov-shift features:

  R1 prototype       : the existing pipeline (class-mean codes + cosine). Baseline.
  Full probe (R4)    : the ridge probe on the full 10000-d code (the C10 reference).
  block_ridge float  : the HDC-native update -- block-diagonal ridge (d^3/B^2 solves)
                       with float decode.
  block_ridge sign   : THE CANDIDATE -- block_ridge update + W quantized to +-1,
                       decode = integer dot (d - 2*Hamming on packed bits).

For each decoder:
  - zero-shot  : fit on CLEAN features (frozen), decode the corrupted val.
  - ceiling    : fit on the corrupted labeled pool (oracle), decode the corrupted val.
  - update_pts_s : pool / update wall-clock.
  - decode_pts_s : val / decode wall-clock.

This is the table for the README: the baseline vs the new method, in both accuracy
(zs + ceiling) and efficiency (throughput).

Usage:
  uv run python robust_diagnostic/probe_method_table_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds snow,wet_ground,fog,crosstalk \
    --n_blocks 20 \
    --out robust_diagnostic/logs/probe_method_table_covshift_ep10.json
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
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

NUM_CLASSES = 17
CONDS_DEFAULT = ['snow', 'wet_ground', 'fog', 'crosstalk']
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

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

# ---------------- decoders ----------------

def block_ridge_fit(codes, lbls, lam, device, n_blocks=20, num_classes=NUM_CLASSES):
    """Block-diagonal ridge: split d into n_blocks, one d/B ridge per block. Returns
    W (d x C) and the accumulate+solve wall-clock."""
    d = codes.shape[1]
    bs = d // n_blocks
    W = torch.zeros(d, num_classes, device=device)
    t_acc = 0.0
    t_solve = 0.0
    for b in range(n_blocks):
        sl = slice(b * bs, (b + 1) * bs) if b < n_blocks - 1 else slice(b * bs, d)
        Xb = codes[:, sl].float().to(device)
        Y = onehot(lbls, num_classes).to(device)
        t0 = time.time()
        S = Xb.T @ Xb
        T = Xb.T @ Y
        t_acc += time.time() - t0
        t0 = time.time()
        db = S.shape[0]
        W[sl] = torch.linalg.solve(S + lam * torch.eye(db, device=device), T)
        t_solve += time.time() - t0
    return W, t_acc + t_solve

def nystrom_fit(codes, lbls, lam, device, m=1000, num_classes=NUM_CLASSES, seed=11):
    """Nystrom-sketch ridge (the Iteration-4 candidate): P in {+1,-1}^{d x m}, each
    m-dim a random +/-1 mix of ALL d HDC dims (holography preserved). Accumulate the
    m x m sketch S_hat = P^T X^T X P and T_hat = P^T X^T Y, solve in m, W = P A.
    Returns W (d x C) and the synchronized accumulate+solve wall-clock."""
    torch.manual_seed(seed)
    d = codes.shape[1]
    P = (torch.rand(d, m) > 0.5).float() * 2 - 1
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    t0 = tic()
    XP = X @ P.to(device)                        # (n, m)
    Shat = XP.T @ XP                             # (m, m)
    That = XP.T @ Y                              # (m, C)
    t_acc = toc(t0)
    t0 = tic()
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    W = P.to(device) @ A                         # (d, C)
    t_solve = toc(t0)
    return W, t_acc + t_solve

def decode_float(codes, W, chunk=100000):
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W.detach().cpu()).argmax(dim=1))
    return torch.cat(preds)

def decode_sign(codes, W, chunk=100000):
    Wq = W.detach().cpu().sign()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ Wq).argmax(dim=1))
    return torch.cat(preds)

def proto_decode(codes, protos, proto_lbls, device, chunk=100000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    for s in range(0, len(codes), chunk):
        hc = F.normalize(codes[s:s + chunk].to(device), p=2, dim=1)
        preds.append(proto_lbls[(hc @ protos.T).argmax(dim=1)].cpu())
    return torch.cat(preds)

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
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--n_blocks", type=int, default=20)
    parser.add_argument("--nystrom_m", type=int, default=1000,
                        help="Nystrom sketch dim for the candidate row (m ~ 1000-2000)")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_method_table_results.json")
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

    # bounded clean sample for the zero-shot probe fit
    max_clean = min(args.max_clean, len(fa))
    ci = torch.randperm(len(fa))[:max_clean]
    fa_s, la_s = fa[ci], la[ci]
    clean_codes = hdc_codes(fa_s, proj, device)
    # zero-shot decoders fit on clean
    lr_zs = LogisticRegression(max_iter=1000, C=1.0)
    lr_zs.fit(clean_codes[:100000].numpy(), la_s[:100000].numpy())
    protos_zs, plbl_zs = build_hdc_prototypes(fa_s, la_s, proj, device=device)

    results = {'label': args.label, 'n_blocks': args.n_blocks, 'conds': {}}
    print(f"\n{'='*118}")
    print(f"=== {args.label}: README method table (pool {args.pool_size}, n_blocks {args.n_blocks}) ===")
    print(f"{'='*118}")

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

        r = {}

        # R1 prototype
        t0 = time.time()
        protos_or, plbl_or = build_hdc_prototypes(pool, pl, proj, device=device)
        r['proto_update_s'] = time.time() - t0
        t0 = time.time()
        r['proto_zs'] = compute_miou(proto_decode(val_codes, protos_zs, plbl_zs, device), vl)
        r['proto_ceiling'] = compute_miou(proto_decode(val_codes, protos_or, plbl_or, device), vl)
        r['proto_decode_s'] = time.time() - t0
        r['proto_decode_pts_s'] = len(val) / r['proto_decode_s']

        # full probe (R4): fit on pool, decode. zs = the frozen clean-fit LR probe.
        W_or = block_ridge_fit(pool_codes, pl, args.lam, device, 1)[0]  # 1 block = full ridge
        r['full_probe_zs'] = compute_miou(
            torch.tensor(lr_zs.predict(val_codes[:100000].numpy())), vl)
        # full probe oracle via a full-block ridge refit on the pool (single block = full d solve)
        r['full_probe_ceiling'] = compute_miou(decode_float(val_codes, W_or), vl)

        # block_ridge candidate: fit on clean (zs) and pool (ceiling)
        t0 = time.time()
        Wb_zs = block_ridge_fit(clean_codes, la_s, args.lam, device, args.n_blocks)[0]
        r['block_fit_zs_s'] = time.time() - t0
        t0 = time.time()
        Wb_or = block_ridge_fit(pool_codes, pl, args.lam, device, args.n_blocks)[0]
        r['block_fit_s'] = time.time() - t0
        r['block_update_pts_s'] = len(pool) / r['block_fit_s']
        r['block_float_zs'] = compute_miou(decode_float(val_codes, Wb_zs), vl)
        r['block_float_ceiling'] = compute_miou(decode_float(val_codes, Wb_or), vl)
        t0 = time.time()
        r['block_sign_zs'] = compute_miou(decode_sign(val_codes, Wb_zs), vl)
        r['block_sign_ceiling'] = compute_miou(decode_sign(val_codes, Wb_or), vl)
        r['block_decode_s'] = time.time() - t0
        r['block_decode_pts_s'] = len(val) / r['block_decode_s']
        r['rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        # Nystrom+sign candidate (Iteration 4): holographic sketch (m ~ 1000-2000),
        # W quantized to +/-1 for the integer popcount decode.
        Wn_zs, tn_zs = nystrom_fit(clean_codes, la_s, args.lam, device, args.nystrom_m)
        Wn_or, tn_or = nystrom_fit(pool_codes, pl, args.lam, device, args.nystrom_m)
        r['nystrom_fit_zs_s'] = tn_zs
        r['nystrom_update_s'] = tn_or
        r['nystrom_update_pts_s'] = len(pool) / tn_or if tn_or > 0 else None
        r['nystrom_float_zs'] = compute_miou(decode_float(val_codes, Wn_zs), vl)
        r['nystrom_float_ceiling'] = compute_miou(decode_float(val_codes, Wn_or), vl)
        t0 = time.time()
        r['nystrom_sign_zs'] = compute_miou(decode_sign(val_codes, Wn_zs), vl)
        r['nystrom_sign_ceiling'] = compute_miou(decode_sign(val_codes, Wn_or), vl)
        r['nystrom_decode_s'] = time.time() - t0
        r['nystrom_decode_pts_s'] = len(val) / r['nystrom_decode_s']

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  {'decoder':<22} {'zero-shot':>9} {'ceiling':>9} {'update_pts/s':>12} {'decode_pts/s':>12}")
        print(f"  {'R1 prototype':<22} {r['proto_zs']:>9.4f} {r['proto_ceiling']:>9.4f} "
              f"{'-':>12} {r['proto_decode_pts_s']:>12,.0f}")
        print(f"  {'full probe (R4)':<22} {r['full_probe_zs']:>9.4f} {r['full_probe_ceiling']:>9.4f} "
              f"{'-':>12} {'-':>12}")
        print(f"  {'block_ridge float':<22} {r['block_float_zs']:>9.4f} {r['block_float_ceiling']:>9.4f} "
              f"{r['block_update_pts_s']:>12,.0f} {r['block_decode_pts_s']:>12,.0f}")
        print(f"  {'block_ridge sign':<22} {r['block_sign_zs']:>9.4f} {r['block_sign_ceiling']:>9.4f} "
              f"{r['block_update_pts_s']:>12,.0f} {r['block_decode_pts_s']:>12,.0f}")
        print(f"  {'Nystrom+sign *':<22} {r['nystrom_sign_zs']:>9.4f} {r['nystrom_sign_ceiling']:>9.4f} "
              f"{r['nystrom_update_pts_s']:>12,.0f} {r['nystrom_decode_pts_s']:>12,.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ THE TABLE ===")
    print("The * row (block_ridge sign) is the candidate for the README: HDC-native")
    print("(block-diagonal ridge at 10000-d, quantized W = integer popcount decode).")
    print("Compare zero-shot/ceiling vs R1 prototype (baseline) and full probe (R4):")
    print("  - does block_ridge sign keep most of the R4 ceiling gain?")
    print("  - is its update_pts/s ~= the prototype's fit rate (efficiency story)?")

if __name__ == "__main__":
    main()
