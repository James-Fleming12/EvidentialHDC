"""seven_cls_diag.py: background diagnostics for the 7-class pivot (~40 min, eval-only).

Runs three checks on the existing class-count models (classcount_seven, all17) to
frame why the 7-class setting could work and how it differs from 17 classes:

  1. PER-CLASS breakdown: clean / fog / crosstalk IoU per class. Shows the 7-class
     mIoU is carried by the well-supported superclasses (road, vegetation, manmade)
     and the 17-class is dragged by the dead rare classes (bicycle, truck, person).
  2. LABEL-FREE GATED PROTOTYPE UPDATE (the TTA thread): zero-shot decode with the
     clean centroids vs a distance-gated weighted re-estimate of the centroids on
     the corrupt points (weight = cosine to the clean centroid), vs the full-label
     oracle. Does updating prototypes when the distance signal says confident beat
     keeping the clean centroids?
  3. ISOTROPY of the clean feature space (reusing the isotropy diagnostics):
     participation ratio, top-5 variance, HDC dead-coordinate fraction, Hamming.
     Is the 7-class encoder space structurally healthier for the HDC pathway?

Usage:
  uv run python robust_diagnostic/seven_cls_diag.py
  uv run python robust_diagnostic/seven_cls_diag.py --models seven
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import yaml
import torch
import torch.nn.functional as F

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import compute_miou, get_hdc_projection
from robust_diagnostic.isotropy_diag import isotropy_metrics

MODELS = {
    'seven': ('robust_diagnostic/logs/classcount_seven', 'config/labels/semantic-kitti-7.yaml'),
    'all17': ('robust_diagnostic/logs/classcount_all17', 'config/labels/semantic-kitti-all.yaml'),
}
CONDS = ['clean', 'fog', 'crosstalk']
GATED_FRACS = [0.5, 0.25, 0.1]

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_features(model, parser, device, num_frames=50):
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
    for c in range(1, 17):
        m = z[l == c]
        if len(m) > 0:
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means

def decode(z, lbls, centroids, device):
    """Nearest-clean-centroid decode -> (mIoU, per-class IoU dict, preds, maxcos)."""
    dev = z.device
    order = sorted(centroids)
    cm = F.normalize(torch.stack([centroids[c] for c in order]), p=2, dim=1).to(dev)
    sims = F.normalize(z, p=2, dim=1) @ cm.T
    maxcos, best = sims.max(dim=1)
    preds = torch.tensor(order, device=dev)[best]
    lbl = lbls.to(dev)
    miou = compute_miou(preds, lbl)
    per_class = {}
    for c in order:
        m = lbl == c
        if m.sum() == 0:
            continue
        tp = int(((preds == c) & m).sum().item())
        fp = int(((preds == c) & ~m).sum().item())
        fn = int(((preds != c) & m).sum().item())
        d = tp + fp + fn
        per_class[c] = tp / d if d > 0 else 0.0
    return miou, per_class, preds, maxcos

def gated_update(z, lbls, centroids, keep_frac, device, use_true=False):
    """Re-estimate centroids on the top-keep_frac points, then decode the FULL set.
    Label-free (use_true=False): points are assigned to a class by the clean-centroid
    decode, and the keep gate is the cosine to the clean centroid. Oracle
    (use_true=True): points are assigned by their TRUE labels. Classes with no kept
    points keep the clean centroid."""
    dev = z.device
    order = sorted(centroids)
    cm = F.normalize(torch.stack([centroids[c] for c in order]), p=2, dim=1).to(dev)
    sims = F.normalize(z, p=2, dim=1) @ cm.T
    maxcos, best = sims.max(dim=1)
    preds = torch.tensor(order, device=dev)[best]
    assign = lbls.to(dev) if use_true else preds
    k = max(1, int(keep_frac * len(z)))
    keep = torch.zeros(len(z), dtype=torch.bool, device=dev)
    keep[torch.topk(maxcos, k).indices] = True

    new_cm = torch.zeros_like(cm)
    cnt = torch.zeros(len(order), device=dev)
    for i, c in enumerate(order):
        sel = keep & (assign == c)
        if sel.sum() > 0:
            new_cm[i] = z[sel].mean(dim=0)
            cnt[i] = sel.sum()
    new_cm = F.normalize(new_cm, p=2, dim=1)
    fallback = cnt == 0
    new_cm[fallback] = cm[fallback]
    new_centroids = {c: new_cm[i] for i, c in enumerate(order)}
    return decode(z, lbls, new_centroids, device)

def oracle_update(z, lbls, centroids, device):
    """Full-label oracle: re-estimate the centroids on the corrupt pool using the
    TRUE labels (all points), then decode."""
    return gated_update(z, lbls, centroids, 1.0, device, use_true=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="seven,all17")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/seven_cls_results.json")
    args = parser.parse_args()

    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    results = {}
    for label in [m.strip() for m in args.models.split(',') if m.strip()]:
        load_path, cfg_path = MODELS[label]
        DATA = yaml.safe_load(open(cfg_path, 'r'))
        print(f"\n{'='*80}\n=== {label} ({load_path}) ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path,
                             method='supcon_vib')
        model = trainer.model

        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        zc, lc = extract_features(model, clean_parser, device, args.frames)
        centroids = class_centroids(zc, lc)

        r = {'classes': sorted(centroids)}

        # check 1 + 2: per-class + gated-update, per condition
        for cond in CONDS:
            if cond == 'clean':
                z, l = zc, lc
            else:
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                z, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            z = z.to(device)
            zs, per_class, _, _ = decode(z, l, centroids, device)
            row = {'zs_miou': float(zs), 'per_class': {str(c): round(v, 4) for c, v in per_class.items()}}
            for frac in GATED_FRACS:
                m, _, _, _ = gated_update(z, l, centroids, frac, device)
                row[f'gated_update_{frac}'] = float(m)
            mo, _, _, _ = oracle_update(z, l, centroids, device)
            row['oracle_update'] = float(mo)
            r[cond] = row
            print(f"  {cond:<10} zs {zs:.4f} | per-class "
                  + " ".join(f"{c}:{v:.2f}" for c, v in per_class.items())
                  + "\n      gated-update " + " ".join(
                      f"{frac}:{row[f'gated_update_{frac}']:.4f}" for frac in GATED_FRACS)
                  + f" | oracle-update {mo:.4f}")

        # check 3: isotropy / HDC-health of the clean feature space (full metrics,
        # including the HDC dead-coordinate fraction over the real Bernoulli projection)
        iso = isotropy_metrics(zc, proj)
        r['clean_isotropy'] = {k: iso[k] for k in
                               ['pr', 'top5_frac', 'log10_cond', 'mean_abs_cos',
                                'mean_frac', 'hdc_dead_frac', 'hdc_hamming']}
        print(f"  clean isotropy: PR {iso['pr']:.1f} | top5 {iso['top5_frac']:.3f} | "
              f"logcond {iso['log10_cond']:.2f} | |cos| {iso['mean_abs_cos']:.3f} | "
              f"deadF {iso['hdc_dead_frac']:.3f} | hamm {iso['hdc_hamming']:.3f}")
        results[label] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
