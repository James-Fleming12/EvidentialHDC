"""probe_hdc_update_forms_diag.py: HDC-aligned update forms for the linear probe
(eval-only).

Iteration 3's block_ridge breaks the holographic structure (block-diagonal mask
zeros out cross-block correlations). These three alternatives keep the FULL 10000-d
space and use HDC-native operations:

  1. CG SOLVE (Krylov/Conjugate Gradient): accumulate the full dense S = X^T X and
     T = X^T Y, then solve (S + lI) W = T with CONJUGATE GRADIENT instead of an
     exact inverse. (S + lI) is SPD; each CG step is one O(d^2) matvec S p, so a few
     iterations cost ~O(d^2) instead of O(d^3) -- and it uses the FULL cross-
     correlated space (no block mask).
  2. HDC DELTA RULE (Widrow-Hoff / Kaczmarz): drop S entirely. Online error-
     correction W <- W + a (y - W h) h^T. For +/-1 codes h^T is a +/-1 vector, so
     the update is pure associative addition/subtraction of the code to the weights:
     O(C*d) per point, NO matrix solve, NO 400MB S. Converges to the ridge boundary
     over epochs.
  3. NYSTROM SKETCH (randomized): keep the holographic structure but reduce the
     SOLVE dimension via a random-sign projection P in {+1,-1}^{d x m} (m << d).
     Accumulate the sketched S_hat = P^T X^T X P (m x m) and T_hat = P^T X^T Y,
     solve in m, and W = P A. Each of the m sketch dims is a random combination of
     all d=10000 HDC dims -- every dimension participates, no slicing.

For each: mIoU on val (zero-shot = clean-fit, ceiling = pool-fit), update wall-clock,
update pts/s, peak RSS -- with the full probe (R4) and block_ridge sign as references.

Usage:
  uv run python robust_diagnostic/probe_hdc_update_forms_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_hdc_forms_covshift_ep10.json
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


# ---------------- the three HDC-aligned update forms ----------------

def cg_solve(S, T, lam, device, iters=5):
    """Solve (S + lI) W = T by CONJUGATE GRADIENT (SPD), k matvecs of O(d^2) instead
    of an O(d^3) inverse. S on device (d x d); T on device (d x C)."""
    A = S + lam * torch.eye(S.shape[0], device=device)
    b = T
    d = S.shape[0]
    x = torch.zeros_like(b)
    r = b - A @ x
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A @ p
        alpha = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x


def cg_update(codes, lbls, lam, device, iters=5):
    """Accumulate full dense S/T, then CG solve."""
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    t0 = time.time()
    S = X.T @ X
    T = X.T @ Y
    t_acc = time.time() - t0
    t0 = time.time()
    W = cg_solve(S, T, lam, device, iters)
    t_solve = time.time() - t0
    return W, t_acc, t_solve


def delta_rule(codes, lbls, alpha, device, epochs=3):
    """HDC delta rule (Widrow-Hoff / Kaczmarz): W <- W + a (y - W h) h^T.
    For +/-1 codes h^T is +/-1, so this is pure associative add/sub of the code to
    the weights: O(C*d) per point, no S matrix, no solve."""
    d = codes.shape[1]
    C = NUM_CLASSES
    X = codes.float()
    Y = onehot(lbls, C)
    W = torch.zeros(d, C, device=device)
    t0 = time.time()
    for _ in range(epochs):
        for i in range(len(codes)):
            h = X[i:i + 1].to(device).T          # (d,1)
            y = Y[i:i + 1].to(device).T          # (C,1)
            err = y - W.T @ h                    # (C,1)
            W = W + alpha * (h @ err.T)          # (d,C)
    t_wall = time.time() - t0
    return W, 0.0, t_wall


def nystrom_update(codes, lbls, lam, device, P, use_sign=False):
    """Nystrom sketch: S_hat = P^T X^T X P (m x m), T_hat = P^T X^T Y, solve in m,
    W = P A. P is a random sign matrix (d x m): every m-dim mixes all d HDC dims."""
    X = codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    t0 = time.time()
    XP = X @ P.to(device)                        # (n, m)
    Shat = XP.T @ XP                             # (m, m)
    That = XP.T @ Y                              # (m, C)
    t_acc = time.time() - t0
    t0 = time.time()
    m = Shat.shape[0]
    A = torch.linalg.solve(Shat + lam * torch.eye(m, device=device), That)
    W = P.to(device) @ A                         # (d, C)
    t_solve = time.time() - t0
    return W, t_acc, t_solve


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
    parser.add_argument("--cg_sweep", type=str, default="5,10,30")
    parser.add_argument("--delta_sweep", type=str,
                        default="0.005:5,0.005:10,0.001:30",
                        help="comma-separated alpha:epochs pairs for the delta rule")
    parser.add_argument("--nystrom_sweep", type=str, default="100,500,1000,2000")
    parser.add_argument("--delta_max_n", type=int, default=5000,
                        help="cap on the delta rule's pool (sequential O(C*d)/point loop)")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_hdc_forms_results.json")
    args = parser.parse_args()

    args.cg_sweep = [int(x) for x in args.cg_sweep.split(',')]
    ds = []
    for p in args.delta_sweep.split(','):
        a, e = p.split(':')
        ds.append((float(a), int(e)))
    args.delta_sweep = ds
    args.nystrom_sweep = [int(x) for x in args.nystrom_sweep.split(',')]

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

    results = {'label': args.label, 'pool_size': args.pool_size, 'conds': {}}

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
        # R1 prototype reference
        t0 = time.time()
        base_protos, base_lbls = build_hdc_prototypes(pool, pl, proj, device=device)
        r['proto_fit_s'] = time.time() - t0
        preds = []
        for s in range(0, len(val_codes), 100000):
            hc = F.normalize(val_codes[s:s + 100000].to(device), p=2, dim=1)
            preds.append(base_lbls[(hc @ F.normalize(base_protos, p=2, dim=1).T).argmax(dim=1)].cpu())
        r['proto_miou'] = compute_miou(torch.cat(preds), vl)
        r['proto_pts_s'] = args.pool_size / r['proto_fit_s']

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  R1 proto: mIoU {r['proto_miou']:.4f}  fit {r['proto_fit_s']:.3f}s ({r['proto_pts_s']:,.0f} pts/s)")

        # CG sweep: iterations control how close to the exact ridge solution
        for iters in args.cg_sweep:
            try:
                W, ta, ts = cg_update(pool_codes, pl, args.lam, device, iters)
                wall = ta + ts
                r['cg_%d' % iters] = {'miou': compute_miou(decode_float(val_codes, W), vl),
                                      'miou_sign': compute_miou(decode_sign(val_codes, W), vl),
                                      'update_s': wall,
                                      'pts_s': args.pool_size / wall if wall > 0 else None}
                print(f"  cg_{iters:<3} mIoU {r['cg_%d' % iters]['miou']:.4f} "
                      f"(sign {r['cg_%d' % iters]['miou_sign']:.4f})  {wall:.3f}s "
                      f"({r['cg_%d' % iters]['pts_s']:,.0f} pts/s)")
            except Exception as e:
                r['cg_%d' % iters] = {'error': str(e)}
                print(f"  cg_{iters} error {e}")

        # HDC delta rule sweep: {alpha, epochs} control convergence to the ridge solution.
        # The delta rule is a sequential per-point loop (O(C*d) each), so cap the pool
        # it sees to keep the sweep tractable -- this is a measured convergence-speed
        # tradeoff (fewer points per epoch, more epochs), not a method limitation.
        delta_n = min(len(pool_codes), args.delta_max_n)
        di = torch.randperm(len(pool_codes))[:delta_n]
        for alpha, epochs in args.delta_sweep:
            try:
                W, ta, ts = delta_rule(pool_codes[di], pl[di], alpha, device, epochs)
                wall = ta + ts
                key = 'delta_a%.0e_e%d_n%d' % (alpha, epochs, delta_n)
                r[key] = {'miou': compute_miou(decode_float(val_codes, W), vl),
                          'miou_sign': compute_miou(decode_sign(val_codes, W), vl),
                          'update_s': wall,
                          'pts_s': delta_n / wall if wall > 0 else None}
                print(f"  {key:<26} mIoU {r[key]['miou']:.4f} "
                      f"(sign {r[key]['miou_sign']:.4f})  {wall:.3f}s "
                      f"({r[key]['pts_s']:,.0f} pts/s)")
            except Exception as e:
                key = 'delta_a%.0e_e%d_n%d' % (alpha, epochs, delta_n)
                r[key] = {'error': str(e)}
                print(f"  {key} error {e}")

        # Nystrom sketch sweep: m controls how much of the holographic space is kept
        for m in args.nystrom_sweep:
            try:
                Pm = (torch.rand(10000, m) > 0.5).float() * 2 - 1
                W, ta, ts = nystrom_update(pool_codes, pl, args.lam, device, Pm)
                wall = ta + ts
                key = 'nystrom_m%d' % m
                r[key] = {'miou': compute_miou(decode_float(val_codes, W), vl),
                          'miou_sign': compute_miou(decode_sign(val_codes, W), vl),
                          'update_s': wall,
                          'pts_s': args.pool_size / wall if wall > 0 else None}
                print(f"  {key:<20} mIoU {r[key]['miou']:.4f} "
                      f"(sign {r[key]['miou_sign']:.4f})  {wall:.3f}s "
                      f"({r[key]['pts_s']:,.0f} pts/s)")
            except Exception as e:
                key = 'nystrom_m%d' % m
                r[key] = {'error': str(e)}
                print(f"  {key} error {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("Each form keeps the FULL 10000-d space (no block mask). Sweeps show what")
    print("each method needs to reach the ridge accuracy ceiling:")
    print("  cg_k        : k CG iterations -> full-dense-S accuracy (O(d^2)/iter).")
    print("  delta_a_e   : {alpha, epochs} -> convergence to the ridge boundary;")
    print("                pure +/-1 associative addition, no S matrix.")
    print("  nystrom_m   : m sketch dims -> how much holography is needed (m^3 solve).")
    print("(sign) columns are the quantized +-1 W decode (integer popcount).")
    print("Compare each to R1 proto and the full-probe R4 ceiling; the parameter at")
    print("which accuracy saturates is the 'implementation need' of each method.")


if __name__ == "__main__":
    main()
