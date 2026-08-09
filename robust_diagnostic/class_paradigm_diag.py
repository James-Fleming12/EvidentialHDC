"""class_paradigm_diag.py: triage diagnostic - does the 7/14-class map lift the recovery
of our previous robust-encoder methods, using ONLY the available weights? (~40 min)

The class map is an eval-time GT aggregation over the model's 128D features, so any
available weight can be evaluated under any target map (7 / 14 / 17): we build clean
centroids from the map-aggregated GT and decode in feature space. The model is loaded
with its OWN training config (so its head/param groups match the checkpoint); the
eval parsers use the target map's config.

Answers:
  - hardneg @ 7 vs @ 17: does the 7-class map lift fog/crosstalk recovery of the
    hard-negative encoder (the strongest artifact-separation method)?
  - hardneg @ 7 vs plain @ 7 (the true trained-7 baseline, classcount_seven): is the
    hardneg encoder's ceiling higher under the same map?
  - plain / hardneg @ 14: does the middle-ground map keep the recovery?

Per (weight, map, condition): zero-shot nearest-centroid mIoU, distance-gated
retained mIoU at 0.25/0.1, correct-vs-wrong AUROC, and the true-label oracle ceiling.

Caveat: a 17-class-trained encoder's features were shaped by 17-class training, so
this is a proxy for what a 7/14-class-TRAINED hardneg would do; the classcount_seven
weights are the true plain-7 baseline.

Usage:
  uv run python robust_diagnostic/class_paradigm_diag.py
  uv run python robust_diagnostic/class_paradigm_diag.py --weights hardneg --maps seven,fourteen
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

# (load_path, method, training config path)
WEIGHTS = {
    'hardneg': ('logs/med_pretrain_supcon_vib_hardneg', 'supcon_vib_hardneg',
                'config/labels/semantic-kitti-all.yaml'),
    'plain_med': ('logs/med_pretrain_supcon_vib', 'supcon_vib',
                  'config/labels/semantic-kitti-all.yaml'),
    'seven_trained': ('robust_diagnostic/logs/classcount_seven', 'supcon_vib',
                      'config/labels/semantic-kitti-7.yaml'),
}
MAPS = {
    'seven': 'config/labels/semantic-kitti-7.yaml',
    'fourteen': 'config/labels/semantic-kitti-14.yaml',
    'all17': 'config/labels/semantic-kitti-all.yaml',
}
CONDS = ['clean', 'fog', 'crosstalk']


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
    for c in range(1, 32):
        m = z[l == c]
        if len(m) > 0:
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means


def decode_gate_oracle(z, lbls, centroids, device):
    """Returns dict: zs mIoU, gated mIoU at 0.25/0.1, AUROC, oracle mIoU."""
    dev = z.device
    order = sorted(centroids)
    cm = F.normalize(torch.stack([centroids[c] for c in order]), p=2, dim=1).to(dev)
    sims = F.normalize(z, p=2, dim=1) @ cm.T
    maxcos, best = sims.max(dim=1)
    preds = torch.tensor(order, device=dev)[best]
    lbl = lbls.to(dev)
    correct = (preds == lbl)
    out = {'zs': float(compute_miou(preds, lbl)), 'auroc': float(roc_auc_score(
        correct.cpu().numpy(), maxcos.cpu().numpy()))}

    # gated retained-set decode
    for frac in (0.25, 0.1):
        k = max(1, int(frac * len(z)))
        keep = torch.zeros(len(z), dtype=torch.bool, device=dev)
        keep[torch.topk(maxcos, k).indices] = True
        out[f'gated{frac}'] = float(compute_miou(preds[keep], lbl[keep]))

    # true-label oracle: re-estimate centroids on ALL corrupt points by true class
    new_cm = torch.zeros_like(cm)
    cnt = torch.zeros(len(order), device=dev)
    for i, c in enumerate(order):
        sel = lbl == c
        if sel.sum() > 0:
            new_cm[i] = z[sel].mean(dim=0)
            cnt[i] = sel.sum()
    new_cm = F.normalize(new_cm, p=2, dim=1)
    new_cm[cnt == 0] = cm[cnt == 0]
    o_sims = F.normalize(z, p=2, dim=1) @ new_cm.T
    o_preds = torch.tensor(order, device=dev)[o_sims.argmax(dim=1)]
    out['oracle'] = float(compute_miou(o_preds, lbl))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="hardneg,plain_med,seven_trained")
    parser.add_argument("--maps", type=str, default="seven,fourteen,all17")
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/class_paradigm_results.json")
    args = parser.parse_args()

    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    wl = [w.strip() for w in args.weights.split(',') if w.strip()]
    ml = [m.strip() for m in args.maps.split(',') if m.strip()]
    results = {}

    for wname in wl:
        load_path, method, train_cfg = WEIGHTS[wname]
        DATA_T = yaml.safe_load(open(train_cfg, 'r'))
        print(f"\n{'='*80}\n=== {wname} ({load_path}, trained with {train_cfg}) ===")
        trainer = GenTrainer(ARCH, DATA_T, args.kitti_dir, load_path, path=load_path, method=method)
        model = trainer.model
        r_w = {}

        for mname in ml:
            DATA_M = yaml.safe_load(open(MAPS[mname], 'r'))
            clean_parser = build_parser(args.kitti_dir, DATA_M, ARCH)
            zc, lc = extract_features(model, clean_parser, device, args.frames)
            centroids = class_centroids(zc, lc)
            r_m = {'classes': len(centroids)}
            print(f"  [{mname} map, {len(centroids)} classes present]")
            for cond in CONDS:
                if cond == 'clean':
                    z, l = zc, lc
                else:
                    cdir = os.path.join(args.kittic_dir, cond, 'heavy')
                    if not os.path.exists(cdir):
                        cdir = os.path.join(args.kittic_dir, cond, 'moderate')
                    z, l = extract_features(model, build_parser(cdir, DATA_M, ARCH), device, args.frames)
                z = z.to(device)
                d = decode_gate_oracle(z, l, centroids, device)
                r_m[cond] = d
                print(f"    {cond:<10} zs {d['zs']:.4f} | gated@0.25 {d['gated0.25']:.4f} "
                      f"@0.1 {d['gated0.1']:.4f} | AUROC {d['auroc']:.3f} | oracle {d['oracle']:.4f}")
            r_w[mname] = r_m
        results[wname] = r_w

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
