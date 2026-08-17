"""al_readiness_diag.py: point-level active-learning readiness across feature
extractors (eval-only, fast).

The Pillar-3 active-learning framework queries ONE point per cluster and labels its
neighborhood. The right readiness metric is therefore POINT-level cluster purity
(nearest-neighbor agreement), not the class-mean probe. For each checkpoint and each
corrupted condition, per class and aggregated, this reports:

  - nn1_purity : fraction of the class's corrupted points whose nearest same-pool
                 neighbor (excluding self) is the same class
  - nnk_purity : fraction whose k-nearest majority is the same class
  - corr_tightness : mean cosine of the class's points to their corrupted class mean
  - oracle     : the labeled-recovery ceiling for the class (HDC, val)

Higher nn1/nnk purity = the active-learning framework needs fewer labels on that
extractor. Higher oracle = the TTA/labeled ceiling is higher.

Usage:
  uv run python robust_diagnostic/al_readiness_diag.py
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
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 weighted_mean_update)

CONDS = ['fog', 'crosstalk']
NUM_CLASSES = 17

# default checkpoints: plain DGLSS++ med, robust DGLSS++ 21ep, soft-anchor blend05 med
CHECKPOINTS = [
    ('dglsspp_med', 'supcon_vib_dglsspp', 'robust_diagnostic/logs/supcon_vib_dglsspp'),
    ('robust_21ep', 'supcon_vib_dglsspp_corsupcon',
     'robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon'),
    ('blend05_med', 'supcon_vib_dglsspp_corsupcon_blend05',
     'robust_diagnostic/logs/med_blend05/supcon_vib_dglsspp_corsupcon_blend05'),
]

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

def clean_class_means(feats, lbls):
    means = {}
    for c in range(1, NUM_CLASSES):
        m = feats[lbls == c]
        if len(m):
            means[c] = F.normalize(m.mean(0), p=2, dim=0)
    return means

def decode_preds(protos, feats, proto_lbls, proj, device, chunk=50000):
    protos = F.normalize(protos, p=2, dim=1)
    preds = []
    for s in range(0, len(feats), chunk):
        hc = F.normalize(torch.sign(feats[s:s + chunk].to(device) @ proj), p=2, dim=1)
        sims = hc @ protos.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

def per_class_iou(preds, lbls, classes):
    out = {}
    for c in classes:
        tp = int(((preds == c) & (lbls == c)).sum().item())
        fp = int(((preds == c) & (lbls != c)).sum().item())
        fn = int(((preds != c) & (lbls == c)).sum().item())
        denom = tp + fp + fn
        out[c] = tp / denom if denom > 0 else 0.0
    return out

def nn_purities(pool, pool_l, classes, k=10, max_pts=20000, chunk=4096):
    """Per-class 1-NN and k-NN same-class purity on (a seeded subsample of) the pool.
    Chunked similarity; self-excluded. Returns {class: (nn1, nnk)} and overall means."""
    torch.manual_seed(0)
    if len(pool) > max_pts:
        idx = torch.randperm(len(pool))[:max_pts]
        pool, pool_l = pool[idx], pool_l[idx]
    zn = F.normalize(pool, p=2, dim=1)
    n = len(pool)
    nn1 = torch.zeros(n, dtype=torch.bool)
    nnk = torch.zeros(n, dtype=torch.bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m = e - s
        sim = zn[s:e] @ zn.T                                    # (m, n)
        sim[torch.arange(m), torch.arange(s, e)] = -1e9          # exclude self
        vals, inds = torch.topk(sim, k, dim=1)
        knn_lbl = pool_l[inds]                                   # (m, k)
        nn1[s:e] = knn_lbl[:, 0] == pool_l[s:e]
        nnk[s:e] = (knn_lbl == pool_l[s:e].unsqueeze(1)).float().mean(dim=1) > 0.5
    rows = {}
    means1, meansk = [], []
    for c in classes:
        m = pool_l == c
        n_c = int(m.sum())
        if n_c == 0:
            rows[c] = {'nn1_purity': float('nan'), 'nnk_purity': float('nan')}
            continue
        p1 = float(nn1[m].float().mean().item())
        pk = float(nnk[m].float().mean().item())
        rows[c] = {'nn1_purity': p1, 'nnk_purity': pk}
        means1.append(p1); meansk.append(pk)
    return rows, (sum(means1) / len(means1) if means1 else float('nan'),
                  sum(meansk) / len(meansk) if meansk else float('nan'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--checkpoints", type=str, default="",
                        help="comma-separated label:method:path triples to compare "
                             "(defaults to the three medium extractors)")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/al_readiness_results.json")
    args = parser.parse_args()

    if args.checkpoints:
        ckpts = []
        for spec in args.checkpoints.split(','):
            parts = spec.strip().split(':')
            if len(parts) != 3:
                raise SystemExit(f"bad checkpoint spec: {spec!r} (want label:method:path)")
            ckpts.append(tuple(parts))
    else:
        ckpts = CHECKPOINTS

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)

    results = {}
    header = (f"{'extractor':<12} {'cond':<10} | {'nn1':>5} {'nnk':>5} | "
              f"{'car or':>6} {'agg or':>6} | {'car nn1':>6} {'car nnk':>6}")
    print(header)
    print('-' * len(header))

    for label, method, path in ckpts:
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model
        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        cmeans = clean_class_means(clean_f, clean_l)
        cids = sorted(cmeans)

        results[label] = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            torch.manual_seed(42)
            perm = torch.randperm(len(f))
            pool, pool_l = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
            val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]

            classes = sorted(cids)
            nn, (m1, mk) = nn_purities(pool, pool_l, classes)
            # tightness
            col_of = {c: j for j, c in enumerate(classes)}
            means_mat = torch.stack([cmeans[c] for c in classes])
            zn = F.normalize(pool, p=2, dim=1)
            cos128 = zn @ means_mat.t()
            tight = {}
            for c in classes:
                m = pool_l == c
                if int(m.sum()) == 0:
                    tight[c] = float('nan'); continue
                tight[c] = float(cos128[m, col_of[c]].mean().item())
            # oracle per class
            def preds(protos):
                return decode_preds(protos, val, proto_lbls, proj, device)
            ones = torch.ones(len(pool), device=device)
            iou_or = per_class_iou(preds(weighted_mean_update(base_protos, proto_lbls, pool,
                                                              pool_l.to(device), ones, proj, device)),
                                   vl, classes)
            for c in classes:
                nn[c]['corr_tightness'] = tight[c]
                nn[c]['oracle'] = iou_or[c]
            agg_or = float(sum(iou_or[c] for c in classes) / len(classes))
            results[label][cond] = {'nn1_mean': m1, 'nnk_mean': mk, 'oracle_mean': agg_or,
                                    'per_class': {str(c): nn[c] for c in classes}}
            car = nn.get(4, {})
            print(f"{label:<12} {cond:<10} | {m1:>5.3f} {mk:>5.3f} | "
                  f"{car.get('oracle', float('nan')):>6.3f} {agg_or:>6.3f} | "
                  f"{car.get('nn1_purity', float('nan')):>6.3f} {car.get('nnk_purity', float('nan')):>6.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
