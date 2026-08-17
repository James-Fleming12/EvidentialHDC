"""probe_geometric_tta_diag.py: label-free GEOMETRIC test-time adaptation for the
linear probe (eval-only, no plots). Abandons T entirely.

Iteration 11 proved the pseudo-label half is poisoned: every gate/weight/soft/
coverage-preserving scheme for T = X^T D_t Y stays at or below no_gate, the wrong
labels anti-align with the oracle rotation, and even a PERFECT-purity T cannot
reproduce the oracle (C/COVERAGE). The uncorrupted half is S = X^T X: the pool
geometry. These methods use ONLY S (and the frozen W_zs) to re-rotate the decoder
toward the oracle, never generating a pseudo-label.

All three work with the EIGENSPACES of the 10000 x 10000 covariances
S_c = X_c^T X_c and S_t = X_t^T X_t, recovered WITHOUT building S: the
randomized-SVD (Halko et al.) eigenspace uses the shared Nystrom sketch
P in {+1,-1}^{d x m} (seed-11, m=1000) to capture range(X), then SVDs the
m x d product; exact when range(XP) contains range(X), and every step is
matrix-free on X (the same accumulate-and-solve machinery as the ridge).

  A. SUBSPACE ALIGNMENT / PROC RUSTES (Fernando et al.): W_new = U_t (U_c^T W_zs)
     with U_c / U_t the top-k eigenspaces of S_c / S_t. If the corruption is a
     pure rotation of the same subspace (U_t = R^T U_c), this is EXACT:
     X_t W_new = X_c W_zs. SVD-rotation variants (t2c / c2t) cover permuted
     basis conventions. Sweeps k in {8, 32, 128, m}.
  B. CORAL covariance alignment (Sun et al.): W_new = S_t^-1/2 S_c^1/2 W_zs on
     the top-k eigenspace (whiten the target covariance, recolor with the clean
     one), plus a whitening-only control (S_t^-1/2, no recoloring).
  C. TRANSDuctive label diffusion (Zhou et al.): the ONLY method that touches
     labels. Point graph G = D^-1/2 A D^-1/2 with A the Hamming similarity of
     the binarized codes, A = (X X^T / d + 1 1^T) / 2 in [0, 1] (d = 10000; D
     from row sums, always positive). Anchor labels Y_sparse = top-1% / top-5%
     by frozen-probe confidence (zero elsewhere), diffuse
     Y_diff = (I - a G)^-1 Y_sparse via matrix-free CG-20
     (A(z) = z - a G z, G z = D^-1/2 ((X (X^T (D^-1/2 z)))/d + sum)/2), then the
     STANDARD full-space ridge (S keeps ALL points, fixing the Iteration-9
     coverage loss). Sweeps a in {0.1, 0.5, 0.9}. Oracle-anchored variant
     (Y_sparse = true labels of the correct points) is the bound.

Diagnostics per condition (JSON + log):
  refs         : frozen / oracle / cos(W_zs, W_oracle) / sketch spectral
                 overlap (singular values of U_c^T U_t: all ~1 = the shift is a
                 pure rotation of the same subspace; decaying = new directions).
  procrustes   : k sweep, both frame conventions, mIoU + cos to oracle.
  coral        : full / rank-k / whiten-only, mIoU + cos to oracle.
  diffusion    : anchor fraction x alpha + oracle-anchored bound.
  controls     : mean-shift decode-time bias (first-order alignment).

Usage:
  uv run python robust_diagnostic/probe_geometric_tta_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/probe_geometric_tta_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import numpy as np
import torch

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'
SKETCH_SEED = 11

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
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(codes), chunk):
        preds.append((codes[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)

def scores(W, codes, chunk=100000):
    W = W.detach().cpu()
    outs = []
    for s in range(0, len(codes), chunk):
        outs.append(codes[s:s + chunk].float() @ W)
    return torch.cat(outs, dim=0)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def cos_sim(Wa, Wb):
    a = Wa.detach().cpu().float().reshape(-1)
    b = Wb.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

# ---------------- sketch-space machinery (shared by all methods) ----------------

def get_sketch(d, m, device):
    """The shared Nystrom sketch P in {+1,-1}^{d x m} (SKETCH_SEED fixed so the
    clean and target covariances live in the same m-dim coordinate frame)."""
    torch.manual_seed(SKETCH_SEED)
    return (torch.rand(d, m, device=device) > 0.5).float() * 2 - 1

def rsvd_eig(Xd, P, k):
    """Randomized-SVD eigenspace of S = X^T X (Halko et al.). The range of X is
    captured by the SVD of XP (n x m), then the SVD of U^T X (m x d) gives the
    right singular vectors exactly when range(XP) contains range(X). Fully
    matrix-free on X. Returns (eigvals k, V d x k orthonormal)."""
    XP = Xd @ P                                   # n x m
    U, _, _ = torch.linalg.svd(XP, full_matrices=False)   # n x m orthonormal
    B = U.t() @ Xd                                # m x d
    _, s, Qt = torch.linalg.svd(B, full_matrices=False)
    V = Qt.t()[:, :k]                             # d x k right singular vectors
    return (s[:k] ** 2), V

def warm_start_factor(Xd, Y, P, lam):
    """The Nystrom warm-start factor A_zs (m x C): the probe in the sketch frame
    (W = P A). The frozen W_zs is the CG-refined version of P A."""
    XP = Xd @ P
    Shat = XP.t() @ XP + lam * torch.eye(P.shape[1], device=P.device)
    That = XP.t() @ Y.float().to(P.device)
    return torch.linalg.solve(Shat, That)

def ridge_fit(Xd, Y, P, lam, iters, device):
    """Full-space ridge: Nystrom warm start + matrix-free CG refinement
    (the established update; S and T both on the same points)."""
    Yd = Y.float().to(device)
    x = P @ warm_start_factor(Xd, Y, P, lam)
    b = Xd.t() @ Yd
    def A(v):
        return Xd.t() @ (Xd @ v)
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

def mw_and_cos(W, val_codes, vl, W_oracle):
    return {'miou': compute_miou(decode(W, val_codes), vl),
            'cos_oracle': cos_sim(W, W_oracle)}

# ---------------- label diffusion on the point graph ----------------

def diffuse_labels(Xd, Y_sparse, alpha, device, iters=20):
    """Y_diff = (I - a G)^-1 Y_sparse on the normalized point graph built from
    the Hamming similarity of the binarized codes:
      A = (X X^T / d + 1 1^T) / 2  in [0, 1]   (d = 10000, 1 = n-vector)
      D_i = row sums of A (always positive),  G = D^-1/2 A D^-1/2.
    Solved by CG with A(z) = z - a G z, G z = D^-1/2 ((X (X^T (D^-1/2 z)))/d
    + sum(D^-1/2 z))/2. All matvecs matrix-free on X."""
    n, d = Xd.shape
    D = (Xd @ (Xd.t() @ torch.ones(n, 1, device=device)) / d + n) / 2  # row sums
    D_inv_sqrt = (D + 1e-8).pow(-0.5).view(-1)
    Yd = Y_sparse.float().to(device)
    def A(z):
        w = D_inv_sqrt.unsqueeze(1) * z
        Aw = (Xd @ (Xd.t() @ w) / d + w.sum(dim=0, keepdim=True).expand(n, -1)) / 2
        Gz = D_inv_sqrt.unsqueeze(1) * Aw
        return z - alpha * Gz
    x = torch.zeros_like(Yd)
    b = Yd
    r = b - A(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=0)
    for _ in range(iters):
        Ap = A(p)
        alpha_k = rs_old / ((p * Ap).sum(dim=0) + 1e-30)
        x = x + alpha_k.unsqueeze(0) * p
        r = r - alpha_k.unsqueeze(0) * Ap
        rs_new = (r * r).sum(dim=0)
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta.unsqueeze(0) * p
        rs_old = rs_new
    return x.float()

# ---------------- main ----------------

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
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--nystrom_m", type=int, default=1000)
    parser.add_argument("--cg_iters", type=int, default=8)
    parser.add_argument("--procrustes_ks", type=str, default="8,32,128,1000")
    parser.add_argument("--coral_ks", type=str, default="128,256,1000")
    parser.add_argument("--diffuse_anchors", type=str, default="0.01,0.05")
    parser.add_argument("--diffuse_alphas", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_geometric_tta_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    p_ks = [int(x) for x in args.procrustes_ks.split(',')]
    c_ks = [int(x) for x in args.coral_ks.split(',')]
    d_anchors = [float(x) for x in args.diffuse_anchors.split(',')]
    d_alphas = [float(x) for x in args.diffuse_alphas.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'conds': {}}

    for cond in conds:
        t_cond = tic()
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

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]
        clean_codes = hdc_codes(fa[ci], proj, device)
        Y_clean = onehot(la[ci], NUM_CLASSES)
        Y_oracle = onehot(pl, NUM_CLASSES)

        P = get_sketch(10000, args.nystrom_m, device)

        # frozen probe on the clean fit (the label-free reference + W_zs)
        Xc = clean_codes.float().to(device)
        W_zs = ridge_fit(Xc, Y_clean, P, args.lam, args.cg_iters, device)
        del clean_codes
        Xd = pool_codes.float().to(device)          # target pool, kept on device

        # oracle and no-gate references (S and T both from the target pool)
        W_oracle = ridge_fit(Xd, Y_oracle, P, args.lam, args.cg_iters, device)

        # pseudo-labels from the frozen probe (only for the diffusion anchors)
        sm = torch.softmax(scores(W_zs, pool_codes), dim=1)
        pconf = sm.max(dim=1).values
        ppred = sm.argmax(dim=1)
        pseudo_correct = (ppred == pl)

        r = {'refs': {}, 'procrustes': {}, 'coral': {}, 'diffusion': {},
             'controls': {}, 'synthesis': []}

        def mw(W):
            return compute_miou(decode(W, val_codes), vl)

        r['refs'] = {
            'frozen': mw(W_zs),
            'oracle': mw(W_oracle),
            'w_zs_cos_oracle': cos_sim(W_zs, W_oracle),
        }

        # ---- eigenspaces of S_c and S_t (randomized SVD, matrix-free on X) ----
        # Spectral overlap: singular values of U_c^T U_t. All ~1 = the shift is a
        # pure rotation of the same subspace; decaying = new directions.
        k_max = min(args.nystrom_m, 512)
        eig_c, U_c = rsvd_eig(Xc, P, k_max)
        eig_t, U_t = rsvd_eig(Xd, P, k_max)
        ov = torch.linalg.svdvals(U_c.t() @ U_t)
        r['refs']['spectral_overlap_sv'] = [float(v) for v in ov[:min(32, len(ov))]]
        r['refs']['eig_c_top'] = [float(v) for v in eig_c[:8]]
        r['refs']['eig_t_top'] = [float(v) for v in eig_t[:8]]

        # ---- A. Subspace alignment / Procrustes (d-dim eigenspaces) ----
        # PLAIN basis match: W_new = U_t (U_c^T W_zs). If the corruption is a pure
        # rotation of the same subspace (overlap ~1), U_t = R^T U_c and this is
        # EXACT: X_t W_new = X_c R (R^T U_c U_c^T W_zs) = X_c W_zs (W_zs in span U_c).
        # The SVD-rotation variants handle permuted / rotated basis conventions.
        for k in p_ks:
            if k > k_max:
                continue
            Uc_k = U_c[:, :k]
            Ut_k = U_t[:, :k]
            W_plain = Ut_k @ (Uc_k.t() @ W_zs)
            r['procrustes'][f'k{k}_plain'] = {'miou': mw(W_plain),
                                              'cos_oracle': cos_sim(W_plain, W_oracle)}
            # t2c: rotate the target basis onto the clean basis
            M = Uc_k.t() @ Ut_k
            U, _, Vt = torch.linalg.svd(M)
            Rk = U @ Vt
            W_t2c = Ut_k @ (Rk @ (Uc_k.t() @ W_zs))
            r['procrustes'][f'k{k}_t2c'] = {'miou': mw(W_t2c),
                                            'cos_oracle': cos_sim(W_t2c, W_oracle)}
            # c2t: rotate the clean basis into the target frame
            W_c2t = Uc_k @ (Rk.t() @ (Ut_k.t() @ W_zs))
            r['procrustes'][f'k{k}_c2t'] = {'miou': mw(W_c2t),
                                            'cos_oracle': cos_sim(W_c2t, W_oracle)}

        # ---- B. CORAL covariance alignment (d-dim eigenbases) ----
        # W_new = S_t^-1/2 S_c^1/2 W_zs with S^+/-1/2 ~= U diag(eig^+/-1/2) U^T
        # on the top-k eigenspace (the whitened-and-recolored probe):
        #   W_new = U_t diag(lt^-1/2) (U_t^T U_c) diag(lc^1/2) U_c^T W_zs.
        for k in c_ks:
            k = min(k, k_max)
            Uc_k = U_c[:, :k]
            Ut_k = U_t[:, :k]
            lc_k = eig_c[:k].clamp(min=1e-8)
            lt_k = eig_t[:k].clamp(min=1e-8)
            core = (Ut_k.t() @ Uc_k) * (lt_k.pow(-0.5).unsqueeze(1) * lc_k.sqrt().unsqueeze(0))
            W_coral = Ut_k @ (core @ (Uc_k.t() @ W_zs))
            r['coral'][f'rank{k}'] = {'miou': mw(W_coral),
                                      'cos_oracle': cos_sim(W_coral, W_oracle)}
        # whitening-only control: M = S_t^-1/2 (no clean recoloring)
        k = min(256, k_max)
        W_w = U_t[:, :k] @ ((U_t[:, :k].t() @ W_zs) /
                            eig_t[:k].clamp(min=1e-8).sqrt().unsqueeze(1))
        r['controls']['whiten_only'] = {'miou': mw(W_w),
                                        'cos_oracle': cos_sim(W_w, W_oracle)}

        # ---- C. label diffusion ----
        for frac in d_anchors:
            n_anch = max(1, int(len(pool) * frac))
            thr = torch.quantile(pconf, 1 - frac)
            anch = pconf >= thr
            Y_sparse = onehot(ppred, NUM_CLASSES) * anch.unsqueeze(1).float()
            for a in d_alphas:
                Y_diff = diffuse_labels(Xd, Y_sparse, a, device)
                W_new = ridge_fit(Xd, Y_diff, P, args.lam, args.cg_iters, device)
                r['diffusion'][f'f{frac}_a{a}'] = {'miou': mw(W_new),
                                                   'cos_oracle': cos_sim(W_new, W_oracle),
                                                   'anchor_prec': float(
                                                       pseudo_correct[anch].float().mean().item())}
        # oracle-anchored upper bound: diffuse the TRUE labels of the correct points
        for a in [0.5, 0.9]:
            Y_sparse = onehot(pl, NUM_CLASSES) * pseudo_correct.unsqueeze(1).float()
            Y_diff = diffuse_labels(Xd, Y_sparse, a, device)
            W_new = ridge_fit(Xd, Y_diff, P, args.lam, args.cg_iters, device)
            r['diffusion'][f'oracle_anch_a{a}'] = {'miou': mw(W_new),
                                                   'cos_oracle': cos_sim(W_new, W_oracle)}

        # ---- mean-shift control: decode-time per-class bias (first-order) ----
        mu_c = Xc.mean(dim=0)
        mu_t = Xd.mean(dim=0)
        b_ms = -((mu_t - mu_c) @ W_zs)
        val_scores = scores(W_zs, val_codes) + b_ms.detach().cpu().unsqueeze(0)
        r['controls']['mean_shift'] = {'miou': compute_miou(val_scores.argmax(dim=1), vl),
                                       'cos_oracle': None,
                                       'note': 'decode-time bias, no W change'}

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.4f} | oracle {r['refs']['oracle']:.4f} | "
              f"cos(Wzs,oracle) {r['refs']['w_zs_cos_oracle']:.3f}")
        print(f"  spectral overlap top-8: " + " ".join(
            f"{v:.3f}" for v in r['refs']['spectral_overlap_sv'][:8]))
        print(f"  PROCRUSTES: " + " ".join(
            f"{k}:{v['miou']:.4f}(cos {v['cos_oracle']:.3f})" for k, v in r['procrustes'].items()))
        print(f"  CORAL: " + " ".join(
            f"{k}:{v['miou']:.4f}(cos {v['cos_oracle']:.3f})" for k, v in r['coral'].items()))
        print(f"  DIFFUSION: " + " ".join(
            f"{k}:{v['miou']:.4f}(cos {v['cos_oracle']:.3f})" for k, v in r['diffusion'].items()))
        print(f"  CONTROLS: whiten {r['controls']['whiten_only']['miou']:.4f} | "
              f"mean_shift {r['controls']['mean_shift']['miou']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("refs: cos(Wzs,oracle) is the rotation to recover (small = large rotation).")
    print("  spectral_overlap: singular values of U_c^T U_t (top-k eigenspaces).")
    print("  All ~1 = the shift is a pure rotation of the same subspace, and the")
    print("  plain basis-match (k*_plain) is EXACT. Decaying = the corruption adds")
    print("  NEW directions a rotation alone cannot span; CORAL's eigenvalue")
    print("  reweighting is then the ceiling.")
    print("A. procrustes k*_plain / t2c / c2t: if mIoU climbs toward oracle as k")
    print("   grows, the corruption IS a subspace rotation and S-only alignment")
    print("   works label-free.")
    print("B. coral rank-k: the eigenvalue-aware alignment; whiten_only control.")
    print("C. diffusion: the only method touching labels (top-K% confident anchors);")
    print("   the ridge keeps S=all so coverage is preserved. oracle_anch is the")
    print("   bound: how far geometry can carry a sparse trusted set.")
    print("If ALL geometric methods stay at frozen: the rotation is NOT encoded in")
    print("the second-order statistics of the pool, and the Pillar-3 handoff is the")
    print("only route. If one climbs: label-free TTA exists for the probe after all.")

if __name__ == "__main__":
    main()
