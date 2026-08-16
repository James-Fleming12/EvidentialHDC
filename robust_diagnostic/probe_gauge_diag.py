"""probe_gauge_diag.py: prototype + separability gauge (eval-only).

The feedback reframes the linear probe as a CALIBRATION mechanism rather than the
decoder: keep the prototype as the cheap O(Cd) primary update, add a tiny k-dim
gauge that measures whether the current prototype geometry is still linearly
separable, and only pay the expensive probe correction (Nystrom refit) when the
gauge says it is worth it. This makes the probe an EXCEPTION HANDLER, not the
default decoder.

Metrics, per condition (fit on the corrupted labeled pool = ceiling):
  prototype acc    : the existing R1 decode (the cheap primary decoder).
  gauge            : a k-dim random-sign sketch g(x) = h(x) P (P in {+1,-1}^{d x k},
                     k = 32/64). A tiny ridge probe is fit on a small reservoir in
                     gauge space. delta_gauge = Acc_tiny - Acc_proto is the signal:
                     ~0 => prototype adequate; >>0 => probe-exploitable structure.
  rank-k correction: W = mu + V A with V in {+1,-1}^{d x k} random directions and
                     A (k x C) fit by ridge on the gauge: a CHEAP boundary-rotation
                     correction (the rank-k idea) instead of the full 10000-d ridge.
  full probe (R4)  : the full ridge probe (the expensive reference / ceiling).

The HEADLINE question: does delta_gauge predict the full-probe gain (R4 - proto)?
If the correlation is high, the gauge can gate the expensive refit: pay Nystrom only
when delta_gauge exceeds a threshold. Report the correlation across conditions.

The gauge is SPARSE by design: only a small sampled reservoir is used to fit it
(not every point), so the per-point cost stays O(Cd) for the prototype update plus
an occasional O(dk) projection on the sampled points.

Usage:
  uv run python robust_diagnostic/probe_gauge_diag.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --conds wet_ground,fog \
    --out robust_diagnostic/logs/probe_gauge_covshift_ep10.json
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

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


def proto_decode(codes, mu, lbls_present, device, chunk=100000):
    """Cosine to the class-mean prototypes (R1). mu: (n_present, d) normalized rows;
    lbls_present: the actual class ids for each row. Decodes into the full label space."""
    preds = []
    mu = F.normalize(mu, p=2, dim=1)
    for s in range(0, len(codes), chunk):
        h = F.normalize(codes[s:s + chunk].float(), p=2, dim=1)
        sims = h @ mu.T
        row_ids = sims.argmax(dim=1)
        preds.append(lbls_present[row_ids])
    return torch.cat(preds)


def ridge_gauge(g_codes, lbls, lam, device):
    """Fit a ridge probe in the k-dim gauge space: A = (G^T G + lI)^{-1} G^T Y (k x C)."""
    G = g_codes.float().to(device)
    Y = onehot(lbls, NUM_CLASSES).to(device)
    k = G.shape[1]
    A = torch.linalg.solve(G.T @ G + lam * torch.eye(k, device=device), G.T @ Y)
    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--reservoir", type=int, default=2000,
                        help="gauge reservoir size (the tiny probe fits on this)")
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--gauge_ks", type=str, default="32,64")
    parser.add_argument("--conds", type=str, default="wet_ground,fog")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/probe_gauge_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    ks = [int(x) for x in args.gauge_ks.split(',')]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    results = {'label': args.label, 'reservoir': args.reservoir, 'conds': {}}
    all_delta = []
    all_full_gain = []

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

        # prototype means (the cheap primary decoder)
        mu, present = [], []
        for c in range(1, NUM_CLASSES):
            m = pl == c
            if int(m.sum().item()) > 0:
                mu.append(pool_codes[m].float().mean(dim=0))
                present.append(c)
        mu = F.normalize(torch.stack(mu), p=2, dim=1)
        present = torch.tensor(present)
        proto_preds = proto_decode(val_codes, mu, present, device)
        proto_miou = compute_miou(proto_preds, vl)

        # full probe reference (R4, the expensive ceiling)
        X = pool_codes.float().to(device)
        Y = onehot(pl, NUM_CLASSES).to(device)
        W_full = torch.linalg.solve(X.T @ X + args.lam * torch.eye(10000, device=device), X.T @ Y)
        full_preds = (val_codes.float() @ W_full.cpu()).argmax(dim=1)
        full_miou = compute_miou(full_preds, vl)
        full_gain = full_miou - proto_miou

        r = {'proto_miou': proto_miou, 'full_probe_miou': full_miou,
             'full_gain': full_gain, 'gauge': {}}

        for k in ks:
            torch.manual_seed(7 + k)
            P = (torch.rand(10000, k) > 0.5).float() * 2 - 1
            # gauge: project a RESERVOIR (small sampled pool) to k-d, fit tiny probe
            ri = torch.randperm(len(pool_codes))[:args.reservoir]
            g_pool = (pool_codes[ri].float() @ P).cpu()
            A = ridge_gauge(g_pool, pl[ri], args.lam, device)
            # tiny probe decode in gauge space on the reservoir-held-out val gauge
            g_val = (val_codes.float() @ P).cpu()
            tiny_preds = (g_val @ A.cpu()).argmax(dim=1)
            tiny_miou = compute_miou(tiny_preds, vl)
            delta_gauge = tiny_miou - proto_miou

            # rank-k correction: W = mu + V A; decode score = proto score + g(x) A
            # proto score matrix over the PRESENT classes only
            score_proto = val_codes.float() @ mu.T          # (n, n_present)
            # gauge correction for the present classes: A columns indexed by class id
            A_present = A.cpu()[:, present]                 # (k, n_present)
            corr_preds = present[(score_proto + g_val @ A_present).argmax(dim=1)]
            corr_miou = compute_miou(corr_preds, vl)

            r['gauge'][str(k)] = {'tiny_miou': tiny_miou, 'delta_gauge': delta_gauge,
                                  'rankk_corr_miou': corr_miou}
            all_delta.append(delta_gauge)
            all_full_gain.append(full_gain)

        results['conds'][cond] = r
        print(f"\n--- {cond} ---")
        print(f"  proto {proto_miou:.4f} | full probe {full_miou:.4f} (gain {full_gain:+.4f})")
        for k in ks:
            g = r['gauge'][str(k)]
            print(f"  k={k:<3} tiny-probe {g['tiny_miou']:.4f}  delta_gauge {g['delta_gauge']:+.4f}  "
                  f"rank-k corr {g['rankk_corr_miou']:.4f}")

    # correlation: does delta_gauge predict full_gain?
    if len(all_delta) >= 3:
        corr = float(np.corrcoef(all_delta, all_full_gain)[0, 1])
        results['gauge_correlates_with_full_gain'] = corr
        print(f"\n=== corr(delta_gauge, full_gain) across conditions = {corr:+.3f} ===")
    else:
        results['gauge_correlates_with_full_gain'] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== READ ===")
    print("The headline question: does delta_gauge (a tiny k-dim probe's advantage over")
    print("the prototype) predict the FULL probe gain (R4 - proto)?")
    print("  - If corr is high: the gauge can gate the expensive Nystrom refit -- pay it")
    print("    only when delta_gauge is large. Prototype stays the cheap default decoder.")
    print("  - rank-k corr: how much of the R4 ceiling the cheap correction W=mu+VA recovers")
    print("    (the boundary-rotation correction without the full 10000-d ridge).")
    print("  - The gauge is sparse (only the reservoir is used), so per-point cost stays O(Cd).")


if __name__ == "__main__":
    main()
