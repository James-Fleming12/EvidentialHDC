"""al_dimension_budget_diag.py: the Iteration-6 dimension test (eval-only, no
plots). Does a smaller code space make labels cheaper?

The estimation argument: the per-class contribution to T is a sum of +-1
vectors in the binarized code space, so the class-centroid estimate has noise
~ 1/sqrt(n_c) per coordinate while the signal is tiny after sign-binarization
spreads the 128-d signal over d dims. Relative estimation error scales as
sqrt(d / n_c): labels per class must scale with d. Iterations 3-5 measured the
consequences (10k-d similarities saturate, S0's 9k perfect labels ~= frozen,
128-d promises reach 0.82-0.93 with 1-2 anchors/class). And the C8 finding says
the labeled ceiling is ~dim-invariant -- so the dim is a path-cheapening lever,
not a ceiling lever.

This diagnostic measures the S0/DIRECT-label budget curve (influence-ranked,
class-floored, no expansion -- the best rung from Iteration 5) across code
dims {128 real-valued features, 512, 1k, 2k, 5k, 10k binarized}: for each dim,
the ridge probe W = (S + lI)^-1 X^T Y with S and T at that dim, decode at that
dim, at budgets {100, 300, 1k, 3k, 10k, 30k, 50k}. References: frozen probe and
oracle at each dim.

Outputs: per condition, per dim, the budget -> mIoU curve + frozen/oracle
references + the crossing point (budget at which the curve passes frozen) and
the approach point (budget at which it reaches 90% of the oracle gap).

Usage:
  uv run python robust_diagnostic/al_dimension_budget_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan \
    --label covshift_ep10 --conds fog,crosstalk \
    --out robust_diagnostic/logs/al_dimension_budget_covshift_ep10.json
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

def onehot(lbls, num_classes):
    y = torch.zeros(len(lbls), num_classes)
    y[torch.arange(len(lbls)), lbls.long()] = 1.0
    return y

def decode(W, X, chunk=100000):
    W = W.detach().cpu()
    preds = []
    for s in range(0, len(X), chunk):
        preds.append((X[s:s + chunk].float() @ W).argmax(dim=1))
    return torch.cat(preds)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def tic():
    sync()
    return time.time()

def toc(t0):
    sync()
    return time.time() - t0

def ridge_fit_soft(X, Y, lam, iters, m, device):
    """Ridge in the GIVEN code space (dim = X.shape[1]). S and T both at that
    dim. For the real-valued 128-d arm, X is the raw features."""
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
    parser.add_argument("--dims", type=str, default="128,512,1000,2000,5000,10000",
                        help="code dims to test; 128 = raw real-valued features")
    parser.add_argument("--budgets", type=str, default="100,300,1000,3000,10000,30000,50000")
    parser.add_argument("--conds", type=str, default="fog,crosstalk,snow,wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_dimension_budget_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    dims = [int(x) for x in args.dims.split(',')]
    budgets = [int(x) for x in args.budgets.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)

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

        mc = min(args.max_clean, len(fa))
        ci = torch.randperm(len(fa))[:mc]

        r = {'refs': {}, 'dims': {}, 'synthesis': []}

        for d in dims:
            if d == 128:
                # real-valued arm: the raw features are the code
                Xc = fa[ci].float()
                Xp = pool.float()
                Xv = val.float()
            else:
                proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=d, device=device)
                Xc = torch.sign(fa[ci].to(device) @ proj).cpu().float()
                Xp = torch.sign(pool.to(device) @ proj).cpu().float()
                Xv = torch.sign(val.to(device) @ proj).cpu().float()

            Yc = onehot(la[ci], NUM_CLASSES)
            W_clean = ridge_fit_soft(Xc, Yc, args.lam, args.cg_iters,
                                     args.nystrom_m, device)
            W_oracle = ridge_fit_soft(Xp, onehot(pl, NUM_CLASSES), args.lam,
                                      args.cg_iters, args.nystrom_m, device)

            def mw(W):
                return compute_miou(decode(W, Xv), vl)

            refs = {'frozen': mw(W_clean), 'oracle': mw(W_oracle)}

            # S0/direct-label budget curve: influence-ranked, class-floored.
            # Influence is computed in the 128-d space (the query signal is
            # representation-agnostic); the labels enter T at the tested dim.
            # Per-point influence via the Nystrom sketch of the 128-d X^T X.
            X128 = pool.float()
            torch.manual_seed(SKETCH_SEED)
            P128 = (torch.rand(X128.shape[1], min(args.nystrom_m, X128.shape[1]),
                               device=device) > 0.5).float() * 2 - 1
            XP128 = X128.to(device) @ P128
            Shat = XP128.t() @ XP128 + args.lam * torch.eye(P128.shape[1], device=device)
            M = torch.linalg.inv(Shat)
            MC = XP128 @ M
            I = (MC.norm(dim=1) * (X128.shape[1] ** 0.5)).cpu()

            classes = sorted(set(pl.tolist()) & set(range(1, NUM_CLASSES)))
            class_max_I = {}
            for c in classes:
                m = pl == c
                if int(m.sum().item()) > 0:
                    class_max_I[c] = float(I[m].max().item())
            order_classes = sorted(classes, key=lambda c: -class_max_I[c])
            qidx = []
            for c in order_classes:
                m = pl == c
                j = int(I[m].argmax().item())
                qidx.append(int(m.nonzero().squeeze(1)[j]))
            qidx = torch.tensor(qidx)
            qlbl = pl[qidx]
            # fill remaining budget by pure influence (dedup)
            rest = torch.ones(len(pool), dtype=torch.bool)
            rest[qidx] = False
            extra = torch.argsort(I[rest], descending=True)

            curve = {}
            for B in budgets:
                B = min(B, len(pool))
                idx = torch.cat([qidx, rest.nonzero().squeeze(1)[extra]])[:B]
                idx = torch.unique(idx)
                Y = torch.zeros(len(pool), NUM_CLASSES)
                Y[idx] = onehot(pl[idx], NUM_CLASSES)
                W = ridge_fit_soft(Xp, Y, args.lam, args.cg_iters, args.nystrom_m,
                                   device)
                curve[str(B)] = {'miou': mw(W), 'n': int(len(idx))}

            r['dims'][str(d)] = {'refs': refs, 'curve': curve}

            # crossing / approach points
            cross = None
            approach = None
            for B, e in curve.items():
                if refs['frozen'] is not None and e['miou'] >= refs['frozen'] and cross is None:
                    cross = int(B)
                gap = refs['oracle'] - refs['frozen']
                if gap > 0 and refs['frozen'] is not None and \
                   e['miou'] >= refs['frozen'] + 0.9 * gap and approach is None:
                    approach = int(B)
            r['dims'][str(d)]['cross_budget'] = cross
            r['dims'][str(d)]['approach_budget'] = approach

            syn = r['synthesis']
            syn.append(f"  dim {d}: frozen {refs['frozen']:.3f} / oracle {refs['oracle']:.3f} | "
                       + " ".join(f"b{B}:{e['miou']:.3f}" for B, e in curve.items()) +
                       f" | cross {cross} approach {approach}")

        results['conds'][cond] = r
        print(f"\n=== {cond} ({(toc(t_cond)):.0f}s) ===")
        for line in r['synthesis']:
            print("  " + line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")

    print("\n=== READ ===")
    print("Per dim (128 real-valued / 512-10k binarized), the S0/direct-label")
    print("  budget curve (influence-ranked, class-floored, NO expansion):")
    print("  - cross_budget: where the curve passes the FROZEN probe")
    print("  - approach_budget: where it reaches 90% of the oracle gap")
    print("If smaller dims cross/approach at 10-100x smaller budgets, the")
    print("  estimation argument (labels ~ d) is confirmed and the dim is the")
    print("  path-cheapening lever. If 128-d does not beat 10k-d at equal")
    print("  budget, the bottleneck is elsewhere (balance/coverage, not dim).")
    print("The oracle per dim also re-checks the C8 ceiling-invariance claim.")

if __name__ == "__main__":
    main()
