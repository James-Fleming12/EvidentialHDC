"""ttagate_diag.py: do the density- and norm-gated prototype updates improve the
fog/crosstalk mIoU? (~1.5-2h, eval-only)

Iteration 2 found strong correct-vs-wrong signals that the earlier TTA battery
never used as update weights: local density (AUROC 0.91 on supcon_vib) and feature
norm (AUROC 0.84-0.87 on the DGLSS / DGLSS++ extractors). This tests them as
weighted-prototype-update gates on fog and crosstalk for each extractor.

The gate weight is the signal itself (higher = more likely correct = more update
weight), and the pseudo-labels are the clean-trained LP. Compared against naive
EMA, the confidence gate, the distance gate, zero-shot, and the oracle.

Usage:
  uv run python robust_diagnostic/ttagate_diag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import yaml
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 compute_miou, weighted_mean_update)

CONDS = ['fog', 'crosstalk']
METHODS = ['supcon_vib', 'supcon_vib_dglss', 'supcon_vib_dglsspp']
ORDER = ['naive', 'conf', 'dist', 'norm_gate', 'dens_gate']


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


def proto_miou(feats, lbls, base_protos, proto_lbls, proj, device):
    feats_d = feats.to(device)
    protos = F.normalize(base_protos, p=2, dim=1)
    sims = []
    for start in range(0, len(feats_d), 50000):
        hc = F.normalize(torch.sign(feats_d[start:start + 50000] @ proj), p=2, dim=1)
        sims.append(hc @ protos.T)
    sims = torch.cat(sims, dim=0)
    return compute_miou(proto_lbls[sims.argmax(dim=1)], lbls.to(device))


def class_centroids(z, l):
    means = {}
    for c in range(1, 32):
        m = z[l == c]
        if len(m) > 0:
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means


def local_density(z, k=20, chunk=8192):
    """Higher = farther from k neighbors (sparser). Chunked. Aligned with input."""
    zn = F.normalize(z, p=2, dim=1)
    dens = torch.zeros(len(z))
    for s in range(0, len(z), chunk):
        e = min(s + chunk, len(z))
        sim = zn[s:e] @ zn.T
        kn = min(k + 1, len(z))
        dens[s:e] = -torch.topk(sim, kn, dim=1).values[:, 1:].mean(dim=1)
    return dens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/ttagate_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    results = {}

    for method in METHODS:
        log_dir = os.path.join(args.log_dir, method)
        print(f"\n{'='*80}\n=== {method}: density / norm gated prototype updates ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=log_dir, method=method)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        clean_means = class_centroids(clean_f, clean_l)
        cm = F.normalize(torch.stack([clean_means[c] for c in sorted(clean_means)]), p=2, dim=1)

        print(f"{'cond':<10} {'zs':>7} " + " ".join(f"{t:>9}" for t in ORDER)
              + f" {'oracle':>8}  gap(norm/dens)")
        r_cond = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)

            torch.manual_seed(42)
            perm = torch.randperm(len(f))
            pool_idx, val_idx = perm[:args.pool_size], perm[-args.val_size:]
            pool = f[pool_idx]
            val, vl = f[val_idx], l[val_idx]

            lp_preds = torch.tensor(clf.predict(pool.numpy())).to(device)
            lp_conf = torch.tensor(clf.predict_proba(pool.numpy()).max(axis=1)).to(device)
            ones = torch.ones(len(pool), device=device)

            # gate weight signals: higher = more likely correct = more weight
            norm_w = pool.norm(p=2, dim=1).to(device)
            dens = local_density(pool)
            dens_w = dens.clamp(min=0).to(device)
            dist_w = (F.normalize(pool, p=2, dim=1) @ cm.T).max(dim=1).values.clamp(min=0).to(device)

            def decode(protos):
                return proto_miou(val, vl, protos, proto_lbls, proj, device)

            zs = decode(base_protos)
            oracle = decode(weighted_mean_update(base_protos, proto_lbls, pool, l[pool_idx].to(device),
                                                 ones, proj, device))
            res = {}
            res['naive'] = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, ones, proj, device))
            res['conf'] = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, lp_conf, proj, device))
            res['dist'] = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, dist_w, proj, device))
            res['norm_gate'] = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, norm_w, proj, device))
            res['dens_gate'] = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, dens_w, proj, device))
            r_cond[cond] = {**res, 'zero_shot': zs, 'oracle': oracle}

            def gap(x):
                return (x - zs) / (oracle - zs) if oracle > zs else float('nan')

            print(f"{cond:<10} {zs:>7.4f} " + " ".join(f"{res[t]:>9.4f}" for t in ORDER)
                  + f" {oracle:>8.4f}  {gap(res['norm_gate']):.2f}/{gap(res['dens_gate']):.2f}")
        results[method] = r_cond

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
