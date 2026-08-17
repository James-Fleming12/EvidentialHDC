"""al_geometry_eval.py: the AL-geometry gate for trained feature extractors
(eval-only, no plots). Measures the feature-space properties that the AL/TTA
bottleneck analysis identified as TRAINABLE, and the AL update itself:

  PROPERTY DIAGNOSTICS (what to train toward):
    intra/inter cosine   : the fat-blob geometry (Iterations 0-8: intra-cos
                           0.62-0.70 is what drives mean-estimation sample
                           complexity, R1-prototype viability, and T-error
                           amplification). Target: intra UP, inter DOWN.
    gain quantiles       : the covariance spectrum (Iterations 8-10: the
                           4-6x ridge-relevant error, the fractional update
                           needing beta<1). Target: gain q99/max DOWN.
    participation rank   : how many directions carry the covariance. Target:
                           higher = flatter spectrum.
    1-NN purity          : the packing (Iteration 0). Target: UP.
    mean-k curve         : the class-mean sample complexity (Iteration 7:
                           cos(mu_hat(k), mu_oracle) for k in {2,8,32}).
                           Target: k=2-8 near the k=32 value (tight balls ->
                           fewer points needed).
    kappa (condition num): largest/smallest eigenvalue of the normalized
                           covariance. Target: DOWN.

  AL UPDATE (the Iteration-10 mechanism, on this extractor):
    frozen / spectral-exact ceiling / 10-COMB (fractional-residual at beta
    0.75, eta 0.1) with the oracle-count T_hat at k=8 means/class (64-72
    labels total) -- the exact Iteration-10 setup, so the trained extractor
    can be compared head-to-head with the base on the AL curve.

Usage:
  uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b <ckpt_dir> --method_b <method> --label_b <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/algeom_gate_<label>.json
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

def cos_sim(a, b):
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))

def ridge_fit_soft(X, Y, lam, iters, m, device):
    X = X.to(device)
    torch.manual_seed(SKETCH_SEED)
    m = min(m, X.shape[1])
    P = (torch.rand(X.shape[1], m, device=device) > 0.5).float() * 2 - 1
    XP = X @ P
    Yd = Y.float().to(device)
    Shat = XP.t() @ XP + lam * torch.eye(m, device=device)
    That = XP.t() @ Yd
    x = P @ torch.linalg.solve(Shat, That)
    b = X.t() @ Yd
    def A(v):
        return X.t() @ (X @ v)
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
    parser.add_argument("--mean_ks", type=str, default="2,8,32")
    parser.add_argument("--mean_repeats", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--mean_k_al", type=int, default=8)
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, required=True)
    parser.add_argument("--method_b", type=str, required=True)
    parser.add_argument("--label_b", type=str, default="algeom")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    mean_ks = [int(x) for x in args.mean_ks.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

    results = {'label': args.label_b, 'method': args.method_b, 'conds': {}}

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

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
        Xc = torch.sign(fa[ci].to(device) @ proj).cpu().float()
        Xp = torch.sign(pool.to(device) @ proj).cpu().float()
        Xv = torch.sign(val.to(device) @ proj).cpu().float()
        Xd = Xp.to(device)
        N = Xp.shape[0]

        W_clean = ridge_fit_soft(Xc, onehot(la[ci], NUM_CLASSES), args.lam,
                                 args.cg_iters, args.nystrom_m, device)
        W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                  args.cg_iters, args.nystrom_m, device)

        def mw(W):
            return compute_miou(decode(W, Xv), vl)

        r = {'refs': {}, 'properties': {}, 'al': {}, 'synthesis': []}
        r['refs'] = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        zn = pool.float()
        zn = zn / (zn.norm(dim=1, keepdim=True) + 1e-8)

        # ---- property diagnostics (128-d, the geometry the losses shape) ----
        pr = r['properties']
        # intra/inter cosine
        means = {}
        for c in classes:
            idx = cls_idx[c]
            if len(idx) < 50:
                continue
            means[c] = zn[idx].mean(dim=0)
        cs = sorted(means)
        cmat = torch.stack([means[c] for c in cs])
        cmat = cmat / (cmat.norm(dim=1, keepdim=True) + 1e-8)
        intra, inter = [], []
        for i, c in enumerate(cs):
            m = zn[cls_idx[c]]
            if len(m) == 0:
                continue
            intra.append(float((m @ means[c]).mean().item()))
            for j in range(i + 1, len(cs)):
                inter.append(float((m @ cmat[j]).mean().item()))
        pr['intra_cos'] = float(np.mean(intra)) if intra else None
        pr['inter_cos'] = float(np.mean(inter)) if inter else None
        pr['separation'] = (pr['intra_cos'] - pr['inter_cos']) if intra and inter else None

        # 1-NN purity (full pool, chunked)
        nn1 = 0
        nn1_den = 0
        for c in classes:
            idx = cls_idx[c]
            if len(idx) < 50:
                continue
            sub = zn[idx].to(device)
            n_c = len(idx)
            nn1_den += n_c
            nn_same = 0
            for s in range(0, n_c, 4096):
                e = min(s + 4096, n_c)
                sim_c = sub[s:e] @ zn.to(device).t()
                sim_c[torch.arange(e - s), idx[s:e]] = -1e9
                nn = sim_c.argmax(dim=1)
                nn_same += int((pl[nn.cpu()] == c).sum().item())
            nn1 += nn_same
        pr['nn1_purity'] = nn1 / nn1_den if nn1_den else None

        # covariance spectrum on the pool features (128-d)
        Sf = (pool.float() - pool.float().mean(dim=0))
        cov = (Sf.t() @ Sf) / (len(pool) - 1)
        eig = torch.linalg.eigvalsh(cov.float()).clamp(min=1e-8)
        pr['kappa'] = float((eig[-1] / eig[0]).item())
        pr['participation_rank'] = float((eig.sum() ** 2 / (eig ** 2).sum()).item())
        pr['eig_top3'] = [float(v) for v in eig.flip(0)[:3]]
        pr['eig_bottom3'] = [float(v) for v in eig[:3]]

        # mean-k sample complexity
        pr['mean_k'] = {}
        for k in mean_ks:
            coss = []
            for c in classes:
                idx = cls_idx[c]
                if len(idx) < max(50, k):
                    continue
                mu_true = zn[idx].mean(dim=0)
                mu_true = mu_true / (mu_true.norm() + 1e-8)
                cc = []
                for rep in range(args.mean_repeats):
                    torch.manual_seed(rep)
                    sub = zn[idx[torch.randperm(len(idx))[:k]]].mean(dim=0)
                    sub = sub / (sub.norm() + 1e-8)
                    cc.append(float((sub * mu_true).sum().item()))
                coss.append(float(np.mean(cc)))
            pr['mean_k'][str(k)] = float(np.mean(coss)) if coss else None

        # ---- the AL update (Iteration-10 mechanism, same setup) ----
        al = r['al']
        # spectral-exact ceiling
        S = (Xd.t() @ Xd).double() / N
        eigS, U = torch.linalg.eigh(S)
        eigS = eigS.float()
        U = U.float()
        lam_hat = args.lam / N
        sig = (eigS + lam_hat).clamp(min=lam_hat)
        T_or = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            T_or[:, c] = Xp[cls_idx[c]].sum(dim=0)
        m0 = pl == 0
        if int(m0.sum().item()) > 0:
            T_or[:, 0] = Xp[m0].sum(dim=0)
        T_or = T_or / N
        Uc = U.to(device)
        sig_d = sig.to(device)
        W_or_spec = (Uc @ ((1.0 / sig_d).unsqueeze(1) * (Uc.t() @ T_or.to(device)))).cpu().float()
        al['ceiling_spec'] = mw(W_or_spec)

        # 10-COMB with oracle counts, k=8 means (the Iteration-10 setup)
        T_h = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            idx = cls_idx[c]
            if len(idx) >= max(50, args.mean_k_al):
                torch.manual_seed(2)
                T_h[:, c] = len(cls_idx[c]) * Xp[
                    idx[torch.randperm(len(idx))[:args.mean_k_al]]].mean(dim=0)
        T_h = T_h / N
        W_beta = (Uc @ (sig_d.pow(-args.beta).unsqueeze(1) *
                        (Uc.t() @ T_h.to(device)))).cpu().float()
        W_comb = W_clean.detach().cpu() + args.eta * (W_beta - W_clean.detach().cpu())
        al['combo_10'] = {'miou': mw(W_comb),
                          'n_labels': args.mean_k_al * len(classes)}
        al['beta'] = args.beta
        al['eta'] = args.eta

        # ---- synthesis ----
        syn = r['synthesis']
        syn.append(f"COND {cond}: frozen {r['refs']['frozen']:.3f} / oracle "
                   f"{r['refs']['oracle']:.3f} / spec-ceil {al['ceiling_spec']:.3f}")
        syn.append(f"  props: intra {pr['intra_cos']:.3f} / inter {pr['inter_cos']:.3f} "
                   f"(sep {pr['separation']:.3f}) | nn1 {pr['nn1_purity']:.3f} | "
                   f"kappa {pr['kappa']:.0f} | part-rank {pr['participation_rank']:.0f}")
        syn.append(f"  mean-k: " + " ".join(
            f"k{k}:{pr['mean_k'][str(k)]:.3f}" for k in mean_ks))
        syn.append(f"  AL 10-COMB: {al['combo_10']['miou']:.3f} "
                   f"({al['combo_10']['n_labels']} labels) vs frozen "
                   f"{r['refs']['frozen']:.3f} / spec-ceil {al['ceiling_spec']:.3f}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in syn:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("properties (what to train toward):")
    print("  intra_cos UP, inter_cos DOWN, separation UP = tighter class balls")
    print("  nn1_purity UP = the packing")
    print("  kappa DOWN, participation_rank UP = flatter covariance spectrum")
    print("  mean_k: k=2-8 near k=32 = fewer labels per class mean")
    print("AL (the Iteration-10 mechanism on this extractor):")
    print("  combo_10 = W_frozen + eta(W_beta - W_frozen) at beta/eta, oracle")
    print("  counts, k=8 means/class (64-72 labels). Compare with the base")
    print("  extractor's algeom gate to see whether the training objective")
    print("  improved the AL curve, the ceiling, or both.")

if __name__ == "__main__":
    main()
