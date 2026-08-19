"""al_betaeta_resweep.py: re-sweep (beta, eta) of the Iteration-10 fractional-
residual AL update on a NEW feature extractor, at the CHEAP label budget.

The Iteration-10 recipe (beta=0.75, eta=0.1) was tuned on the base corsupcon
extractor's STEEPER covariance spectrum. The AL-geometry objectives (ball/spec)
flattened it (participation rank 3-5 vs 2-3), which moves the optimal gain
shaping. This evaluates W = W_frozen + eta*(W_beta - W_frozen), W_beta =
(S/N + l/N I)^(-beta) (T_hat/N), across a (beta, eta) grid at k=8 means/class
(64-72 labels total -- the cheap regime), per condition, on an arbitrary
checkpoint. Eval-only, minutes per condition.

Output JSON per condition:
  frozen, oracle (cg), oracle_spec (spectral-exact ceiling),
  budget grid: combo[k=8][beta][eta] -> {miou, w_cos, delta_from_frozen},
  and the per-condition best (beta, eta) at k=8 and its delta vs the default
  beta=0.75/eta=0.1.

Usage:
  uv run python robust_diagnostic/al_betaeta_resweep.py \
    --path_b <ckpt_dir> --method_b <method> --label <label> \
    --conds fog,crosstalk,snow,wet_ground \
    --out robust_diagnostic/logs/al_betaeta_med_<label>.json
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


def mw(W, Xv, vl):
    return compute_miou(decode(W, Xv), vl)


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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tic():
    sync()
    return time.time()


def toc(t0):
    sync()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    ap.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--val_size", type=int, default=100000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--max_clean", type=int, default=200000)
    ap.add_argument("--nystrom_m", type=int, default=1000)
    ap.add_argument("--cg_iters", type=int, default=8)
    ap.add_argument("--mean_k", type=int, default=8)
    ap.add_argument("--betas", type=str, default="0.0,0.25,0.5,0.6,0.75,0.85,1.0")
    ap.add_argument("--etas", type=str, default="0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    ap.add_argument("--path_b", type=str, required=True)
    ap.add_argument("--method_b", type=str, required=True)
    ap.add_argument("--label", type=str, default="med")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    betas = [float(x) for x in args.betas.split(',')]
    etas = [float(x) for x in args.etas.split(',')]
    k = args.mean_k

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

    results = {'label': args.label, 'method': args.method_b, 'mean_k': k,
               'betas': betas, 'etas': etas, 'conds': {}}

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

        r = {'refs': {}, 'combo': {}}
        r['refs']['frozen'] = mw(W_clean, Xv, vl)
        r['refs']['oracle'] = mw(W_oracle, Xv, vl)

        classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
        cls_idx = {c: (pl == c).nonzero().squeeze(1) for c in classes}

        # spectral-exact ceiling (S/N + l/N I)^-1 (T_oracle/N)
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
        UtT_or = Uc.t() @ T_or.to(device)
        W_or_spec = (Uc @ ((1.0 / sig_d).unsqueeze(1) * UtT_or)).cpu().float()
        r['refs']['oracle_spec'] = mw(W_or_spec, Xv, vl)

        # imperfect T at the cheap budget: oracle counts x random-k means
        T_h = torch.zeros(10000, NUM_CLASSES)
        for c in classes:
            idx = cls_idx[c]
            if len(idx) >= max(50, k):
                torch.manual_seed(2)
                T_h[:, c] = len(cls_idx[c]) * Xp[
                    idx[torch.randperm(len(idx))[:k]]].mean(dim=0)
        T_h = T_h / N
        UtT_h = Uc.t() @ T_h.to(device)

        # the 10-COMB grid at k=8
        Wf = W_clean.detach().cpu()
        grid = {}
        for beta in betas:
            grid[str(beta)] = {}
            W_beta = (Uc @ (sig_d.pow(-beta).unsqueeze(1) * UtT_h)).cpu().float()
            for eta in etas:
                W = Wf + eta * (W_beta - Wf)
                grid[str(beta)][str(eta)] = {
                    'miou': mw(W, Xv, vl),
                    'w_cos': cos_sim(W, W_or_spec),
                    'delta': mw(W, Xv, vl) - r['refs']['frozen']}
        r['combo'] = grid

        # best (beta, eta) at this k, and the default-0.75/0.1 reference
        best = max(
            ((beta, eta, v) for beta in betas for eta in etas for v in [grid[str(beta)][str(eta)]]),
            key=lambda x: x[2]['delta'])
        r['best'] = {'beta': best[0], 'eta': best[1], **best[2]}
        r['default'] = grid['0.75']['0.1']

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        print(f"  frozen {r['refs']['frozen']:.3f} / oracle {r['refs']['oracle']:.3f} "
              f"/ spec-ceil {r['refs']['oracle_spec']:.3f}")
        print(f"  default (b=0.75,e=0.1): {r['default']['delta']:+.3f} "
              f"(miou {r['default']['miou']:.3f})")
        print(f"  BEST at k={k}: b={best[0]} e={best[1]} -> {best[2]['delta']:+.3f} "
              f"(miou {best[2]['miou']:.3f})")
        print(f"  best grid deltas:")
        for beta in betas:
            row = " ".join(f"e={e}:{grid[str(beta)][str(e)]['delta']:+.3f}" for e in etas)
            print(f"    b={beta}: {row}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("At k=8 (64-72 labels total): does any (beta, eta) give a POSITIVE delta")
    print("on snow/wet_ground (the conditions frozen-negative in Iteration-10), or")
    print("a materially better delta than the default b=0.75/e=0.1? If the best")
    print("beta/eta differs from 0.75/0.1, the feature-space flattening moved the")
    print("AL optimum and the extractor changes the recipe -- worth continuing to")
    print("train. If snow/wet is negative across the whole grid, the negative-AL")
    print("is a property of the space (or the T_hat/rare-class issue), not a")
    print("(beta, eta) mismatch.")


if __name__ == "__main__":
    main()
