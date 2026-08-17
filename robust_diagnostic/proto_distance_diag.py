"""proto_distance_diag.py: can distance-to-prototype uncertainty recover fog/crosstalk
with the 7-class models? (~30 min, eval-only on existing checkpoints)

Phase 24.12 showed the 7-class models have a much healthier HDC decode (clean proto
mIoU 0.444 -> 0.632) but fog/crosstalk still collapse (proto mIoU ~0.10 / ~0.15).
This diagnostic tests the natural uncertainty lever: distance to the nearest CLEAN
class prototype (128D cosine, scale-invariant). If the retained-set mIoU climbs
toward the clean level as we retain only near-prototype points, then a label-free
distance gate recovers the corruption; if it stays flat, the corruption destroys the
distance structure itself.

Reports per model (seven, all17) and condition (clean, fog, crosstalk):
  - zero-shot nearest-centroid mIoU (100% retention)
  - retained-set mIoU at retention bands 0.9 / 0.75 / 0.5 / 0.25 / 0.1
  - correct-vs-wrong AUROC of the distance signal
  - the full-label oracle (centroids re-estimated on the corrupt pool) as the ceiling

Usage:
  uv run python robust_diagnostic/proto_distance_diag.py
  uv run python robust_diagnostic/proto_distance_diag.py --models seven
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import yaml
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import compute_miou

MODELS = {
    'seven': ('robust_diagnostic/logs/classcount_seven', 'config/labels/semantic-kitti-7.yaml'),
    'all17': ('robust_diagnostic/logs/classcount_all17', 'config/labels/semantic-kitti-all.yaml'),
}
CONDS = ['clean', 'fog', 'crosstalk']
RETENTIONS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]

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
    """Normalized 128D per-class means."""
    means = {}
    for c in range(1, 17):
        m = z[l == c]
        if len(m) > 0:
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means

def decode_and_gate(z, lbls, centroids, retention):
    """Nearest-clean-centroid decode, optionally gated to the top-'retention' by
    highest cosine (smallest distance). Returns (mIoU, correct_mask, maxcos)."""
    dev = z.device
    order = sorted(centroids)
    cm = F.normalize(torch.stack([centroids[c] for c in order]), p=2, dim=1).to(dev)
    sims = F.normalize(z, p=2, dim=1) @ cm.T                      # (N, C)
    maxcos, best = sims.max(dim=1)
    preds = torch.tensor(order, device=dev)[best]
    correct = (preds == lbls.to(dev))                              # (N,) bool
    if retention >= 1.0:
        keep = torch.ones(len(z), dtype=torch.bool, device=dev)
    else:
        k = max(1, int(retention * len(z)))
        keep = torch.zeros(len(z), dtype=torch.bool, device=dev)
        keep[torch.topk(maxcos, k).indices] = True
    miou = compute_miou(preds[keep], lbls.to(dev)[keep])
    return miou, correct, maxcos, keep

def oracle_miou(z, lbls, centroids, device):
    """Full-label oracle: re-estimate centroids on the corrupt points, decode."""
    order = sorted(centroids)
    cm = torch.zeros(len(order), z.shape[1], device=device)
    cnt = torch.zeros(len(order), device=device)
    for i, c in enumerate(order):
        m = lbls.to(device) == c
        if m.sum() > 0:
            cm[i] = z.to(device)[m].mean(dim=0)
            cnt[i] = m.sum()
    cm = F.normalize(cm, p=2, dim=1)
    sims = F.normalize(z.to(device), p=2, dim=1) @ cm.T
    preds = torch.tensor(order, device=device)[sims.argmax(dim=1)]
    return compute_miou(preds, lbls.to(device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="seven,all17")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/proto_distance_results.json")
    args = parser.parse_args()

    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model_labels = [m.strip() for m in args.models.split(',') if m.strip()]
    results = {}
    for label in model_labels:
        load_path, cfg_path = MODELS[label]
        DATA = yaml.safe_load(open(cfg_path, 'r'))
        print(f"\n{'='*80}\n=== {label} ({load_path}) ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, load_path, path=load_path,
                             method='supcon_vib')
        model = trainer.model

        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        zc, lc = extract_features(model, clean_parser, device, args.frames)
        centroids = class_centroids(zc, lc)
        print(f"  clean n {len(zc)}, classes {sorted(centroids)}")

        r = {'clean_centroids': sorted(centroids)}
        for cond in CONDS:
            if cond == 'clean':
                z, l = zc, lc
            else:
                cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                if not os.path.exists(cdir):
                    cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                z, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            z = z.to(device)
            rows = {}
            for ret in RETENTIONS:
                miou, correct, maxcos, keep = decode_and_gate(z, l, centroids, ret)
                rows[f'ret{ret:.2f}'] = float(miou)
            auroc = float(roc_auc_score(correct.cpu().numpy(), maxcos.cpu().numpy()))
            rows['auroc'] = auroc
            rows['oracle'] = float(oracle_miou(z, l, centroids, device))
            r[cond] = rows
            print(f"  {cond:<10} zs {rows['ret1.00']:.4f} | gated "
                  + " ".join(f"{k}:{v:.3f}" for k, v in rows.items() if k.startswith('ret') and k != 'ret1.00')
                  + f" | AUROC {auroc:.3f} | oracle {rows['oracle']:.4f}")
        results[label] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
