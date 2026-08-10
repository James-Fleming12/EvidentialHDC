"""scale_gap_diag.py: per-class autopsy of why the label-free (naive EMA) prototype
update closes far less of the labeled ceiling gap on the MEDIUM DGLSS++ extractor
than on the MICRO one.

Iteration 4 measured the aggregate: DGLSS++ fog naive-EMA closes ~0.20 of the
oracle gap at medium scale vs reaching the ceiling at micro scale. This script
decomposes that difference class-by-class to test the working hypothesis:

  at medium scale the dataset leans so heavily toward majority classes that the
  minority-class features are less robust and sit farther from their class
  prototype, so the label-free weighted update helps them less.

For each scale (micro checkpoint and medium checkpoint) and each collapsed
condition (fog, crosstalk), it computes, PER CLASS on the pooled corrupted
features:

  - freq            : class count in the pool (the class prior)
  - feat_cos        : mean 128D cosine of the class's points to the CLEAN class mean
                      (how close the features sit to the prototype)
  - hdc_cos         : mean cosine of the class's HDC sign codes to the clean prototype
  - zs_correct      : fraction of the class's points that zero-shot decode correctly
  - lp_recall       : fraction of the class's points the logistic probe assigns to it
                      (the pseudo-label quality the naive update depends on)
  - zs / naive / oracle per-class IoU on the held-out val split
  - gap_closed      : (naive - zs) / (oracle - zs)

plus the aggregate mIoU / gap-closed and Spearman correlations of the per-class
numbers against class frequency, to see whether the minority classes are the ones
where the medium-scale update loses ground.

Usage:
  uv run python robust_diagnostic/scale_gap_diag.py
  uv run python robust_diagnostic/scale_gap_diag.py --frames 150 --pool_size 100000
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
                                 weighted_mean_update)

CONDS = ['fog', 'crosstalk']
NUM_CLASSES = 17

MICRO_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp_micro'
MED_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'


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


def spearman(xs, ys):
    """Spearman rho (ties-averaged). Returns (rho, nan)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 4:
        return float('nan')
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = (sum((rx[i] - mx) ** 2 for i in range(n))) ** 0.5
    dy = (sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return (cov / (dx * dy)) if dx * dy > 0 else float('nan')


def pool_per_class_stats(pool, pool_l, lp_preds, preds_pool, means_mat, classes):
    """Per-class feature-robustness stats on the pooled corrupted features."""
    col_of = {c: j for j, c in enumerate(classes)}
    zn = F.normalize(pool, p=2, dim=1)
    cos128 = zn @ means_mat                          # (n, n_classes)
    rows = {}
    for c in classes:
        m = pool_l == c
        n = int(m.sum())
        if n == 0:
            rows[c] = {'freq': 0, 'feat_cos': float('nan'), 'zs_correct': float('nan'),
                       'lp_recall': float('nan')}
            continue
        rows[c] = {
            'freq': n,
            'feat_cos': float(cos128[m, col_of[c]].mean().item()),
            'zs_correct': float((preds_pool[m] == c).float().mean().item()),
            'lp_recall': float((lp_preds[m] == c).float().mean().item()),
        }
    return rows


def hdc_cos_to_own(pool, pool_l, base_protos, proto_lbls, proj, device, chunk=50000):
    """Per-class mean cosine of a class's points to its own HDC prototype."""
    proto_map = {int(c): k for k, c in enumerate(proto_lbls.tolist())}
    sums, counts = {}, {}
    for s in range(0, len(pool), chunk):
        m = pool_l[s:s + chunk] > 0
        if not m.any():
            continue
        z = pool[s:s + chunk][m].to(device)
        l = pool_l[s:s + chunk][m]
        hc = F.normalize(torch.sign(z @ proj), p=2, dim=1)
        for c in l.unique().tolist():
            k = proto_map.get(c)
            if k is None:
                continue
            cc = hc[l == c] @ base_protos[k]
            sums[c] = sums.get(c, 0.0) + float(cc.sum().item())
            counts[c] = counts.get(c, 0) + int((l == c).sum().item())
    return {c: (sums[c] / counts[c] if counts[c] else float('nan')) for c in sums}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--micro_path", type=str, default=MICRO_PATH,
                        help="micro DGLSS++ checkpoint dir (default: the backed-up micro)")
    parser.add_argument("--med_path", type=str, default=MED_PATH,
                        help="medium DGLSS++ checkpoint dir (default: the 24ep/100% run)")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/scale_gap_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    scales = [('micro', args.micro_path), ('med', args.med_path)]
    all_rows = {}

    for name, path in scales:
        print(f"\n{'='*80}\n=== DGLSS++ {name} scale ({path}) ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path,
                             method='supcon_vib_dglsspp')
        model = trainer.model

        print("Extracting clean...")
        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        cmeans = clean_class_means(clean_f, clean_l)
        cids = sorted(cmeans)
        means_mat = F.normalize(torch.stack([cmeans[c] for c in cids]), p=2, dim=1).to(device)

        all_rows[name] = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            print(f"Extracting {cond}...")
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)

            torch.manual_seed(42)
            perm = torch.randperm(len(f))
            pool_idx, val_idx = perm[:args.pool_size], perm[-args.val_size:]
            pool = f[pool_idx]
            val, vl = f[val_idx], l[val_idx]
            pool_l = l[pool_idx]

            lp_preds = torch.tensor(clf.predict(pool.numpy())).to(device)
            ones = torch.ones(len(pool), device=device)
            vd = val.to(device)

            classes = sorted(cids)
            preds_pool = decode_preds(base_protos, pool, proto_lbls, proj, device)
            feat = pool_per_class_stats(pool, pool_l, lp_preds.cpu(), preds_pool,
                                        means_mat, classes)
            hdcc = hdc_cos_to_own(pool, pool_l, base_protos, proto_lbls, proj, device)
            for c in classes:
                feat[c]['hdc_cos'] = hdcc.get(c, float('nan'))

            naive = weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, ones,
                                         proj, device)
            oracle = weighted_mean_update(base_protos, proto_lbls, pool,
                                          pool_l.to(device), ones, proj, device)

            def preds(protos):
                return decode_preds(protos, val, proto_lbls, proj, device)

            p_zs = preds(base_protos)
            p_na = preds(naive)
            p_or = preds(oracle)
            iou_zs = per_class_iou(p_zs, vl, classes)
            iou_na = per_class_iou(p_na, vl, classes)
            iou_or = per_class_iou(p_or, vl, classes)

            for c in classes:
                feat[c]['zs_iou'] = iou_zs[c]
                feat[c]['naive_iou'] = iou_na[c]
                feat[c]['oracle_iou'] = iou_or[c]
                zs, na, orc = iou_zs[c], iou_na[c], iou_or[c]
                feat[c]['gap_closed'] = ((na - zs) / (orc - zs)) if orc > zs else float('nan')

            def mean_iou(d):
                vs = [d[c] for c in classes if d[c] == d[c]]
                return sum(vs) / len(vs) if vs else float('nan')

            zs_m, na_m, orc_m = mean_iou(iou_zs), mean_iou(iou_na), mean_iou(iou_or)
            gap = (na_m - zs_m) / (orc_m - zs_m) if orc_m > zs_m else float('nan')
            all_rows[name][cond] = {'aggregate': {'zs_miou': zs_m, 'naive_miou': na_m,
                                                  'oracle_miou': orc_m, 'gap_closed': gap},
                                    'per_class': {str(c): feat[c] for c in classes}}

            print(f"  aggregate {cond}: zs {zs_m:.4f} naive {na_m:.4f} oracle {orc_m:.4f} "
                  f"gap-closed {gap:.2f}")

    # ---- print per-condition comparison tables ----
    for cond in CONDS:
        print(f"\n{'='*80}\n=== {cond}: per-class, micro vs medium ===\n{'='*80}")
        classes = sorted(all_rows['micro'][cond]['per_class'].keys(), key=int)
        rows = {name: all_rows[name][cond]['per_class'] for name in ['micro', 'med']}
        agg = {name: all_rows[name][cond]['aggregate'] for name in ['micro', 'med']}
        print(f"aggregate gap-closed: micro {agg['micro']['gap_closed']:.2f}  "
              f"med {agg['med']['gap_closed']:.2f}")
        print(f"{'cls':>3} {'freq_m':>7} {'freq_M':>7} | {'fcos_m':>6} {'fcos_M':>6} "
              f"| {'zscor_m':>6} {'zscor_M':>6} {'lprec_m':>6} {'lprec_M':>6} "
              f"| {'zs_m':>5} {'zs_M':>5} {'na_m':>5} {'na_M':>5} {'orc_m':>5} {'orc_M':>5} "
              f"| {'gap_m':>5} {'gap_M':>5}")
        for c in classes:
            a, b = rows['micro'][c], rows['med'][c]
            g = lambda k, d: float('nan') if d.get(k) is None or d[k] != d[k] else d[k]
            print(f"{int(c):>3} {int(a['freq']):>7} {int(b['freq']):>7} | "
                  f"{g('feat_cos', a):>6.3f} {g('feat_cos', b):>6.3f} | "
                  f"{g('zs_correct', a):>6.3f} {g('zs_correct', b):>6.3f} "
                  f"{g('lp_recall', a):>6.3f} {g('lp_recall', b):>6.3f} | "
                  f"{g('zs_iou', a):>5.3f} {g('zs_iou', b):>5.3f} "
                  f"{g('naive_iou', a):>5.3f} {g('naive_iou', b):>5.3f} "
                  f"{g('oracle_iou', a):>5.3f} {g('oracle_iou', b):>5.3f} | "
                  f"{g('gap_closed', a):>5.2f} {g('gap_closed', b):>5.2f}")

    # ---- correlations against class frequency ----
    print(f"\n{'='*80}\n=== correlation with class frequency (Spearman rho) ===\n{'='*80}")
    corrs = {}
    for cond in CONDS:
        for name in ['micro', 'med']:
            pc = all_rows[name][cond]['per_class']
            pairs = [(pc[str(c)]['freq'], pc[str(c)]['feat_cos'], pc[str(c)]['gap_closed'])
                     for c in pc
                     if pc[str(c)]['feat_cos'] == pc[str(c)]['feat_cos']
                     and pc[str(c)]['gap_closed'] == pc[str(c)]['gap_closed']]
            freq = [p[0] for p in pairs]
            fcos = [p[1] for p in pairs]
            gaps = [p[2] for p in pairs]
            r_fcos = spearman(freq, fcos)
            r_gap = spearman(freq, gaps)
            corrs[f'{name}_{cond}'] = {'freq_vs_feat_cos': r_fcos, 'freq_vs_gap_closed': r_gap}
            print(f"{name:<5} {cond:<10} rho(freq, feat_cos)={r_fcos:+.2f}   "
                  f"rho(freq, gap_closed)={r_gap:+.2f}")

    all_rows['correlations'] = corrs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_rows, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
