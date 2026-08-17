"""probe_rep_scan_diag.py: efficiency levers for the linear probe, HDC-native first
(eval-only).

The probe at d=10000 is ~7-8x slower than the R1 prototype update. This scan finds
cheaper ways to compute the SAME ridge update -- staying inside the HDC space and
exploiting the binarization.

The key HDC-native fact: for +/-1 codes, diag(X^T X) = n for EVERY coordinate
(since x_i^2 = 1). So the diagonal-only ridge
    W_c = X^T Y_c / (diag(X^T X) + l) = (n_c mu_c) / (n + l)  proportional to mu_c
is EXACTLY the class-mean prototype (up to scale). This proves two things:
  (a) the probe's entire gain over the prototype is the OFF-DIAGONAL covariance
      (the rotation, Iteration 0), and
  (b) the HDC binarization gives a closed-form, prototype-equivalent bound for free.

Section A -- HDC-NATIVE methods (keep the 10000-d code, use the binarization):
  1. R1 prototype reference (the existing pipeline).
  2. DIAGONAL-RIDGE bound: must equal the prototype mIoU (validates (a)/(b)).
  3. DUAL (Woodbury) W = X^T (X X^T + lI_n)^{-1} Y with the inversion in the sample
     dim n (n << d): the HDC-native efficient update for pooled/chunked updates.
     G = X X^T is a matrix of +/-1 dot products (integers): exact, and hardware-
     friendly (popcount/Hamming on packed bits in a real system).
  4. RLS (Sherman-Morrison) for point-by-point streaming, O(d^2)/pt, no solve.

Section B -- DIMENSION CHECK, framed as the paper's claim about the projection:
The goal is NOT to shrink the projection -- the efficiency win comes from USING the
binarization (Section A). The dimension check settles a DIFFERENT question: is the
probe's linear-separability gain a property of the LARGE 10000-d projection, or of
the BINARIZED GEOMETRY? If the probe keeps its mIoU at a reduced code dim d' or a
second random projection to k, then the large projection dimension never helped --
the gain is the binarized geometry, not the projection size. That is a paper
statement that keeps the method HDC (we keep 10000-d + binarization in the method;
the check just shows the projection size was never the source of the power):
  5. JL second projection of the 10000-d code to k in {128,256,512}: mIoU + cost.
  6. Direct code dim d' in {256,512,1000,2000,5000}: mIoU + cost.

For each: mIoU on val, fit wall-clock, fit throughput (pts/s), peak RSS, at a fixed
pool (default 10k, near the saturated AL budget).

Usage:
  uv run python robust_diagnostic/probe_rep_scan_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_rep_scan_covshift_ep10.json
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

def hdc_codes_at(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y

def predict(W, codes):
    W = W.detach().cpu()
    return (codes.float() @ W).argmax(dim=1)

# ---------------- HDC-native methods (Section A) ----------------

def diagonal_ridge(codes, lbls, lam, num_classes=NUM_CLASSES):
    """W_c = X^T Y_c / (diag(X^T X) + l). For +/-1 codes diag = n * 1, so this is the
    class-mean prototype up to scale. The binarization-simplified bound."""
    X = codes.float()
    Y = onehot(lbls, num_classes)
    t0 = time.time()
    diag = (X * X).sum(dim=0)          # = n for every coordinate
    T = X.T @ Y
    W = T / (diag.unsqueeze(1) + lam)
    return W, time.time() - t0

def dual_woodbury(codes, lbls, lam, device, num_classes=NUM_CLASSES):
    """W = X^T (X X^T + lI_n)^{-1} Y. Inversion in the sample dim n. G = X X^T is
    an integer (+/-1 dot product) matrix -- exact, HDC-native."""
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    t0 = time.time()
    G = X @ X.T                       # n x n
    t_acc = time.time() - t0
    t0 = time.time()
    n = X.shape[0]
    A = torch.linalg.solve(G + lam * torch.eye(n, device=device), Y)
    W = X.T @ A
    t_solve = time.time() - t0
    return W, t_acc, t_solve

def rls(codes, lbls, lam, device, num_classes=NUM_CLASSES):
    """Sherman-Morrison streaming: maintain P = (S + lI)^{-1}, O(d^2) per point."""
    d = codes.shape[1]
    P = torch.eye(d, device=device) / lam
    W = torch.zeros(d, num_classes, device=device)
    t0 = time.time()
    for i in range(len(codes)):
        h = codes[i:i + 1].float().to(device).T
        Ph = P @ h
        denom = 1.0 + (h.T @ Ph).item()
        P = P - (Ph @ Ph.T) / denom
        err = (onehot(lbls[i:i + 1], num_classes).to(device).T - W.T @ h)
        W = W + (P @ h) @ err.T
    return W, 0.0, time.time() - t0

# ---------------- Section B: does the projection help? (ablation) ----------------

def ridge_primal_fit(codes, lbls, lam, device, num_classes=NUM_CLASSES):
    X = codes.float().to(device)
    Y = onehot(lbls, num_classes).to(device)
    d = X.shape[1]
    t0 = time.time()
    S = X.T @ X
    T = X.T @ Y
    t_acc = time.time() - t0
    t0 = time.time()
    W = torch.linalg.solve(S + lam * torch.eye(d, device=device), T)
    return W, t_acc, time.time() - t0

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
    parser.add_argument("--dims", type=str, default="256,512,1000,2000,5000")
    parser.add_argument("--jl_ks", type=str, default="128,256,512")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_rep_scan_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    dims = [int(x) for x in args.dims.split(',')]
    jl_ks = [int(x) for x in args.jl_ks.split(',')]
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    D = fa.shape[1]

    torch.manual_seed(7)
    jl_mats = {k: (torch.randn(10000, k) * 0.5).to(device) for k in jl_ks}

    results = {'label': args.label, 'feature_dim': D, 'pool_size': args.pool_size,
               'conds': {}}

    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]

        proj10 = get_hdc_projection(dim_in=D, dim_out=10000, device=device)
        pool10 = hdc_codes_at(pool, proj10, device)
        val10 = hdc_codes_at(val, proj10, device)

        # --- Section A: HDC-native (10000-d code) ---
        sec_a = {}
        # R1 prototype reference
        t0 = time.time()
        base_protos, base_lbls = build_hdc_prototypes(pool, pl, proj10, device=device)
        proto_fit = time.time() - t0
        t0 = time.time()
        preds = []
        for s in range(0, len(val10), 100000):
            hc = F.normalize(val10[s:s + 100000].to(device), p=2, dim=1)
            preds.append(base_lbls[(hc @ F.normalize(base_protos, p=2, dim=1).T).argmax(dim=1)].cpu())
        proto_decode = time.time() - t0
        sec_a['proto'] = {'miou': compute_miou(torch.cat(preds), vl),
                          'fit_s': proto_fit, 'decode_s': proto_decode,
                          'fit_pts_s': args.pool_size / proto_fit}

        # diagonal-ridge bound (must equal prototype)
        Wd, tw = diagonal_ridge(pool10, pl, args.lam)
        sec_a['diag_ridge'] = {'miou': compute_miou(predict(Wd, val10), vl),
                               'wall_s': tw, 'pts_s': args.pool_size / tw if tw > 0 else None}

        # dual Woodbury (the HDC-native pooled update)
        W, ta, ts = dual_woodbury(pool10, pl, args.lam, device)
        wall = ta + ts
        sec_a['dual_woodbury'] = {'miou': compute_miou(predict(W, val10), vl),
                                  'accum_s': ta, 'solve_s': ts, 'wall_s': wall,
                                  'pts_s': args.pool_size / wall if wall > 0 else None}

        # RLS streaming (small pool only)
        if args.pool_size <= 2000:
            W, tra, trs = rls(pool10, pl, args.lam, device)
            wall = tra + trs
            sec_a['rls'] = {'miou': compute_miou(predict(W, val10), vl),
                            'accum_s': tra, 'solve_s': trs, 'wall_s': wall,
                            'pts_s': args.pool_size / wall if wall > 0 else None}
        else:
            sec_a['rls'] = {'skipped': 'n>2000'}

        # --- Section B: does the projection help? (ablation) ---
        sec_b = {}
        # JL second projection of the 10000-d code to k
        for k in jl_ks:
            t0 = time.time()
            pc_k = torch.sign(pool10.to(device) @ jl_mats[k]).cpu()
            vc_k = torch.sign(val10.to(device) @ jl_mats[k]).cpu()
            W, ta, ts = ridge_primal_fit(pc_k, pl, args.lam, device)
            wall = time.time() - t0 + ta + ts
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            sec_b['jl_%d' % k] = {'miou': compute_miou(predict(W, vc_k), vl),
                                  'accum_s': ta, 'solve_s': ts, 'wall_s': wall,
                                  'pts_s': args.pool_size / wall if wall > 0 else None,
                                  'rss_mb': rss}
        # direct reduced code dim d'
        for d in dims:
            proj = get_hdc_projection(dim_in=D, dim_out=d, device=device)
            pc = hdc_codes_at(pool, proj, device)
            vc = hdc_codes_at(val, proj, device)
            W, ta, ts = ridge_primal_fit(pc, pl, args.lam, device)
            wall = ta + ts
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            sec_b['code_%d' % d] = {'miou': compute_miou(predict(W, vc), vl),
                                    'accum_s': ta, 'solve_s': ts, 'wall_s': wall,
                                    'pts_s': args.pool_size / wall if wall > 0 else None,
                                    'rss_mb': rss}

        results['conds'][cond] = {'sec_A_hdc_native': sec_a, 'sec_B_proj_ablation': sec_b}

        print(f"\n{'='*110}")
        print(f"=== {args.label} [{cond}]  pool {args.pool_size}  feature dim {D} ===")
        print(f"{'='*110}")
        print(f"  SECTION A -- HDC-native (10000-d code):")
        print(f"    {'method':<18} {'mIoU':>7} {'wall_s':>8} {'pts/s':>11}")
        for name, r in sec_a.items():
            if 'miou' not in r:
                continue
            print(f"    {name:<18} {r['miou']:>7.4f} {r.get('wall_s', r.get('fit_s', 0)):>8.3f} "
                  f"{r.get('pts_s', 0):>11,.0f}")
        print(f"  SECTION B -- does the projection help? (ablation):")
        print(f"    {'rep':<18} {'mIoU':>7} {'wall_s':>8} {'pts/s':>11}")
        for name, r in sec_b.items():
            print(f"    {name:<18} {r['miou']:>7.4f} {r['wall_s']:>8.3f} {r['pts_s']:>11,.0f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ THE SCAN ===")
    print("SECTION A (HDC-native, stays at 10000-d -- the method):")
    print("  diag_ridge must equal proto mIoU (for +/-1 codes diag=n, so diag-ridge is")
    print("  the prototype) -- this PROVES the probe's gain is the off-diagonal covariance.")
    print("  dual_woodbury is the HDC-native pooled update (inversion in n, integer G).")
    print("  rls is the streaming form (no solve). Compare pts/s vs proto.fit.")
    print("SECTION B (dimension check -- the paper claim about the PROJECTION, not the")
    print("method): the efficiency win is USING the binarization (Section A). This check")
    print("asks whether the large 10000-d projection itself was the source of the gain.")
    print("  If jl_k / code_d keep the probe mIoU at small k/d', the projection SIZE")
    print("  never helped -- the gain is the binarized geometry. We keep 10000-d +")
    print("  binarization in the method; this just shows the size was never the power.")

if __name__ == "__main__":
    main()
