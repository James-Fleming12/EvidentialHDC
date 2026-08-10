"""gate_structure_diag.py: can any gate signal, or any structure, separate the
update-helpful from the update-harmful points on the DGLSS / DGLSS++ extractors?
(~1h, eval-only)

Iteration 1 showed the weighted prototype updates (naive, confidence, distance)
are near-identical on all three extractors, which means those weight signals are
weakly discriminating. This diagnostic tests whether ANY signal, or a structure
in the feature space, separates points whose updates would help from those that
would harm, on each extractor and condition.

Per extractor (supcon_vib, supcon_vib_dglss, supcon_vib_dglsspp) and condition
(fog, crosstalk, snow control), on a subsample of the frozen 128D features:

  - per-signal correct-vs-wrong AUROC for: LP confidence, entropy, distance to the
    nearest clean prototype, feature norm, top-2 margin, and LOCAL DENSITY
    (mean distance to k neighbors; lower = denser);
  - a logistic-regression fusion AUROC over all signals (does ANY combination
    separate correct from wrong?);
  - a (confidence, distance) grid of the correct fraction, to see whether
    confident-correct and confident-incorrect points occupy separate regions;
  - the local recoverability of the confident-but-wrong points (fraction whose true
    class is in the top-k clean prototypes), the structure active learning could
    exploit.

Usage:
  uv run python robust_diagnostic/gate_structure_diag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from robust_diagnostic.d3ctta_diag import feature_recoverability

CONDS = ['fog', 'crosstalk', 'snow']
METHODS = ['supcon_vib', 'supcon_vib_dglss', 'supcon_vib_dglsspp']


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


def class_centroids(z, l):
    means = {}
    for c in range(1, 32):
        m = z[l == c]
        if len(m) > 0:
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means


def local_density(z, k=20, chunk=8192, max_pts=50000):
    """Mean distance to k nearest neighbors (lower = denser). Chunked, subsampled."""
    if len(z) > max_pts:
        idx = torch.randperm(len(z))[:max_pts]
        z = z[idx]
    zn = F.normalize(z, p=2, dim=1)
    dens = torch.zeros(len(z))
    for s in range(0, len(z), chunk):
        e = min(s + chunk, len(z))
        sim = zn[s:e] @ zn.T
        kn = min(k + 1, len(z))
        topk = torch.topk(sim, kn, dim=1).values
        dens[s:e] = -topk[:, 1:].mean(dim=1)   # negate cosine, higher = farther
    return dens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--max_pts", type=int, default=200000)
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/gate_structure_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    results = {}
    for method in METHODS:
        log_dir = os.path.join(args.log_dir, method)
        print(f"\n{'='*80}\n=== {method}: gate-signal structure ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=log_dir, method=method)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        clean_means = class_centroids(clean_f, clean_l)
        cm = F.normalize(torch.stack([clean_means[c] for c in sorted(clean_means)]), p=2, dim=1)

        header = (f"{'cond':<10} " + " ".join(f"{s:>7}" for s in
                  ['conf', 'entr', 'dist', 'norm', 'marg', 'dens', 'fusion', 'recCW'])
                  + "   C/W gap")
        print(header)
        r_cond = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            if len(f) > args.max_pts:
                idx = torch.randperm(len(f))[:args.max_pts]
                f, l = f[idx], l[idx]

            proba = clf.predict_proba(f.numpy())
            conf = proba.max(axis=1)
            entr = -(proba * np.log(np.clip(proba, 1e-9, 1))).sum(axis=1)
            preds = proba.argmax(axis=1)
            correct = (preds == l.numpy())

            zn = F.normalize(f, p=2, dim=1)
            sims = zn @ cm.T
            dist = sims.max(dim=1).values.numpy()
            margin = (sims.topk(2, dim=1).values[:, 0] - sims.topk(2, dim=1).values[:, 1]).numpy()
            norm = f.norm(p=2, dim=1).numpy()
            dens = local_density(f, max_pts=50000).numpy()

            sigs = {'conf': conf, 'entr': entr, 'dist': dist,
                    'norm': norm, 'margin': margin, 'density': dens}
            aurocs = {s: (roc_auc_score(correct, v) if correct.std() > 0 and v.std() > 0
                          else float('nan')) for s, v in sigs.items()}
            X = np.stack([(v - v.mean()) / (v.std() + 1e-9) for v in sigs.values()], axis=1)
            try:
                lr = LogisticRegression(max_iter=1000).fit(X, correct)
                aurocs['fusion'] = roc_auc_score(correct, lr.decision_function(X))
            except Exception:
                aurocs['fusion'] = float('nan')

            # confident-correct vs confident-wrong: centroid separation in signal space
            th = np.quantile(conf, 0.5)
            cc = np.stack([dist, norm, margin, dens])[:, (conf >= th) & correct]
            cw = np.stack([dist, norm, margin, dens])[:, (conf >= th) & ~correct]
            gap = float(np.linalg.norm(cc.mean(axis=1) - cw.mean(axis=1)))

            # local recoverability of the confident-but-wrong points
            cw_idx = torch.nonzero(torch.tensor((conf >= th) & ~correct)).squeeze(1)
            fr = feature_recoverability(f[cw_idx], l[cw_idx], clean_means) if len(cw_idx) > 1000 \
                else {'rec_of_wrong': float('nan')}

            r_cond[cond] = {'aurocs': {k: float(v) for k, v in aurocs.items()},
                            'cw_gap': gap, 'rec_conf_wrong': fr['rec_of_wrong']}
            print(f"{cond:<10} " + " ".join(f"{aurocs[s]:>7.3f}" for s in
                  ['conf', 'entr', 'dist', 'norm', 'margin', 'density', 'fusion'])
                  + f"  {fr['rec_of_wrong']:>6.3f}   {gap:.3f}")
        results[method] = r_cond

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
