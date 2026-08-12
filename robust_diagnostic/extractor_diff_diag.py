"""extractor_diff_diag.py: per-class feature-structure comparison between two feature
extractors (e.g. plain DGLSS++ medium vs the robust DGLSS++ variant), to answer three
questions directly:

  1. WHAT made crosstalk TTA better?  (which classes' label-free update improved, and
     which per-class feature changes drove it)
  2. WHAT made fog worse?             (which classes' recoverable ceiling dropped, and
     their structure under fog)
  3. WHAT could raise the fog ceiling back?  (is the fog ceiling capped by the SupCon
     clean-anchoring OVER-ALIGNING the corrupted features, measured as per-class
     feat_cos / direction-retention vs the per-class oracle gain)

Per checkpoint, per condition (fog, crosstalk), per class it reports:
  - feat_cos         : mean 128D cosine of the class's corrupted points to the CLEAN
                       class mean (how close to the clean prototype / how much the
                       corruption shift was pulled back toward clean)
  - dir_retention    : cosine(corrupted class mean, clean class mean) -- how much the
                       class DIRECTION survives corruption (1 = unshifted, <1 = shifted)
  - corr_tightness   : mean cosine of corrupted points to the CORRUPTED class mean
                       (intra-class packing under corruption)
  - clean_tightness  : mean cosine of CLEAN points to the clean class mean
  - lp_recall        : logistic-probe recall for the class on the corrupted pool
  - zs / naive / oracle per-class IoU on val, plus oracle_gain = oracle - zs

Usage:
  uv run python robust_diagnostic/extractor_diff_diag.py
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

PATH_A = 'robust_diagnostic/logs/supcon_vib_dglsspp'               # plain DGLSS++ medium
PATH_B = 'robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon'  # robust 21ep


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


def class_structure(pool, pool_l, clean_f, clean_l, clean_means, classes):
    """Per-class feature-structure on the pooled corrupted features + clean tightness."""
    col_of = {c: j for j, c in enumerate(classes)}
    means_mat = torch.stack([clean_means[c] for c in classes])          # (K, 128) normalized
    zn = F.normalize(pool, p=2, dim=1)
    cos128 = zn @ means_mat.t()                                          # (n, K)
    rows = {}
    for c in classes:
        m = pool_l == c
        n = int(m.sum())
        if n == 0:
            rows[c] = {'freq': 0, 'feat_cos': float('nan'),
                       'dir_retention': float('nan'), 'corr_tightness': float('nan'),
                       'clean_tightness': float('nan')}
            continue
        pts = zn[m]
        corr_mean = F.normalize(pts.mean(0), p=2, dim=0)
        rows[c] = {
            'freq': n,
            'feat_cos': float(cos128[m, col_of[c]].mean().item()),
            'dir_retention': float((corr_mean * means_mat[col_of[c]]).sum().item()),
            'corr_tightness': float((pts @ corr_mean).mean().item()),
        }
    for c in classes:
        m = clean_l == c
        if int(m.sum()) == 0:
            rows[c]['clean_tightness'] = float('nan')
            continue
        rows[c]['clean_tightness'] = float(
            (F.normalize(clean_f[m], p=2, dim=1) @ means_mat[col_of[c]]).mean().item())
    return rows


def branch_structure(pool, pool_l, clean_f, clean_l, clean_means, classes, inv_ch):
    """Two-branch decoupling check (Iteration-15 gate): split the concatenated
    bottleneck at channel inv_ch into the invariant slice [0:inv_ch] and the corruption
    slice [inv_ch:] and measure the per-class structure ON EACH SLICE SEPARATELY:
      - inv slice feat_cos       : how anchored the invariant branch is (should stay
                                   HIGH -- the SupCon clean-anchor lands here)
      - corr slice dir_retention : whether the corruption branch retains the shifted
                                   direction (should be < 1 -- the recoverable shift
                                   survives, unlike the full-vector over-alignment)
      - corr slice tightness     : intra-corruption packing of the corr branch
    Returns dict per class: {inv_feat_cos, corr_feat_cos, inv_dir_retention,
    corr_dir_retention, corr_tightness}."""
    out = {}
    for name, sl in (('inv', slice(0, inv_ch)), ('corr', slice(inv_ch, None))):
        rows = class_structure(pool[:, sl], pool_l, clean_f[:, sl], clean_l,
                               {c: F.normalize(m[sl], p=2, dim=0) for c, m in clean_means.items()},
                               classes)
        for c in classes:
            out.setdefault(c, {})[f'{name}_feat_cos'] = rows[c]['feat_cos']
            out[c][f'{name}_dir_retention'] = rows[c]['dir_retention']
            out[c][f'{name}_tightness'] = rows[c]['corr_tightness']
    return out


def cluster_al_stats(pool, pool_l, classes):
    """Active-learning readiness: per class, the fraction of the class's CORRUPTED
    points whose nearest (normalized 128D) CORRUPTED class-mean is their OWN class.
    This is the accuracy of the Pillar-3 'query one point per cluster, label the whole
    cluster' scheme at the class-mean level: high purity = one query per class labels
    most of the class. Also returns the overall mean over present classes."""
    zn = F.normalize(pool, p=2, dim=1)
    means = {}
    for c in classes:
        m = pool_l == c
        if int(m.sum()) > 0:
            means[c] = F.normalize(zn[m].mean(0), p=2, dim=0)
    cids = [c for c in classes if c in means]
    if not cids:
        return {str(c): float('nan') for c in classes}, float('nan')
    means_mat = torch.stack([means[c] for c in cids])
    sims = zn @ means_mat.t()
    preds = torch.tensor(cids, device=pool.device)[sims.argmax(1)]
    out = {}
    vals = []
    for c in classes:
        m = pool_l == c
        if int(m.sum()) == 0:
            out[str(c)] = float('nan')
            continue
        p = float((preds[m] == c).float().mean().item())
        out[str(c)] = p
        vals.append(p)
    return out, (sum(vals) / len(vals) if vals else float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--path_a", type=str, default=PATH_A,
                        help="checkpoint A (default: plain DGLSS++ medium)")
    parser.add_argument("--path_b", type=str, default=PATH_B,
                        help="checkpoint B (default: robust DGLSS++ 21ep)")
    parser.add_argument("--method_a", type=str, default="supcon_vib_dglsspp")
    parser.add_argument("--method_b", type=str, default="supcon_vib_dglsspp_corsupcon")
    parser.add_argument("--label_a", type=str, default="dglsspp_med")
    parser.add_argument("--label_b", type=str, default="robust_21ep")
    parser.add_argument("--inv_ch", type=int, default=0,
                        help="two-branch split point: invariant channels [0:inv_ch], corruption "
                             "channels [inv_ch:]. When >0, also measures the per-branch structure "
                             "(inv/corr feat_cos, dir_retention, tightness) -- the decoupling gate.")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/extractor_diff_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    ckpts = [('A', args.label_a, args.method_a, args.path_a),
             ('B', args.label_b, args.method_b, args.path_b)]
    data = {}

    for tag, label, method, path in ckpts:
        print(f"\n{'='*80}\n=== {label} ({method}, {path}) ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, path, path=path, method=method)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        proj = get_hdc_projection(dim_in=clean_f.shape[1], dim_out=10000, device=device)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        cmeans = clean_class_means(clean_f, clean_l)
        cids = sorted(cmeans)

        data[label] = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)

            torch.manual_seed(42)
            perm = torch.randperm(len(f))
            pool_idx, val_idx = perm[:args.pool_size], perm[-args.val_size:]
            pool, pool_l = f[pool_idx], l[pool_idx]
            val, vl = f[val_idx], l[val_idx]
            lp_preds = torch.tensor(clf.predict(pool.numpy())).to(device)
            ones = torch.ones(len(pool), device=device)

            classes = sorted(cids)
            struct = class_structure(pool, pool_l, clean_f, clean_l, cmeans, classes)
            if args.inv_ch and clean_f.shape[1] > args.inv_ch:
                br = branch_structure(pool, pool_l, clean_f, clean_l, cmeans, classes, args.inv_ch)
                for c in classes:
                    struct[c].update(br.get(c, {}))
            al_purity, al_mean = cluster_al_stats(pool, pool_l, classes)
            for c in classes:
                m = pool_l == c
                struct[c]['lp_recall'] = float((lp_preds[m] == c).float().mean().item()) if m.sum() else float('nan')
                struct[c]['al_purity'] = al_purity.get(str(c), float('nan'))

            def preds(protos):
                return decode_preds(protos, val, proto_lbls, proj, device)

            iou_zs = per_class_iou(preds(base_protos), vl, classes)
            iou_na = per_class_iou(preds(weighted_mean_update(base_protos, proto_lbls, pool,
                                                              lp_preds, ones, proj, device)), vl, classes)
            iou_or = per_class_iou(preds(weighted_mean_update(base_protos, proto_lbls, pool,
                                                              pool_l.to(device), ones, proj, device)), vl, classes)
            for c in classes:
                struct[c]['zs_iou'] = iou_zs[c]
                struct[c]['naive_iou'] = iou_na[c]
                struct[c]['oracle_iou'] = iou_or[c]
                struct[c]['oracle_gain'] = iou_or[c] - iou_zs[c]

            def mean_iou(d):
                vs = [d[c] for c in classes if d[c] == d[c]]
                return sum(vs) / len(vs) if vs else float('nan')

            zs_m, na_m, orc_m = mean_iou(iou_zs), mean_iou(iou_na), mean_iou(iou_or)
            gap = (na_m - zs_m) / (orc_m - zs_m) if orc_m > zs_m else float('nan')
            data[label][cond] = {'aggregate': {'zs': zs_m, 'naive': na_m, 'oracle': orc_m,
                                               'gap_closed': gap, 'al_purity': al_mean},
                                 'per_class': {str(c): struct[c] for c in classes}}
            print(f"  aggregate {cond}: zs {zs_m:.4f} naive {na_m:.4f} oracle {orc_m:.4f} "
                  f"gap {gap:.2f}  one-label-per-cluster purity {al_mean:.3f}")

    # ---- comparison printout ----
    la, lb = args.label_a, args.label_b
    for cond in CONDS:
        print(f"\n{'='*80}\n=== {cond}: per-class, {la} vs {lb} ===\n{'='*80}")
        classes = sorted(data[la][cond]['per_class'].keys(), key=int)
        a, b = data[la][cond]['per_class'], data[lb][cond]['per_class']
        ga, gb = data[la][cond]['aggregate'], data[lb][cond]['aggregate']
        print(f"aggregate: {la} zs {ga['zs']:.4f} naive {ga['naive']:.4f} oracle {ga['oracle']:.4f} "
              f"gap {ga['gap_closed']:.2f} | {lb} zs {gb['zs']:.4f} naive {gb['naive']:.4f} "
              f"oracle {gb['oracle']:.4f} gap {gb['gap_closed']:.2f}")
        print(f"{'cls':>3} | {'fcA':>5} {'fcB':>5} {'dirA':>5} {'dirB':>5} {'tightA':>5} {'tightB':>5} "
              f"{'clnA':>5} {'clnB':>5} {'lprA':>5} {'lprB':>5} | {'zsA':>5} {'zsB':>5} "
              f"{'naA':>5} {'naB':>5} {'orA':>5} {'orB':>5} {'gA':>5} {'gB':>5}")
        for c in classes:
            ra, rb = a[c], b[c]
            g = lambda k, d: float('nan') if d.get(k) is None or d[k] != d[k] else d[k]
            print(f"{int(c):>3} | {g('feat_cos',ra):>5.2f} {g('feat_cos',rb):>5.2f} "
                  f"{g('dir_retention',ra):>5.2f} {g('dir_retention',rb):>5.2f} "
                  f"{g('corr_tightness',ra):>5.2f} {g('corr_tightness',rb):>5.2f} "
                  f"{g('clean_tightness',ra):>5.2f} {g('clean_tightness',rb):>5.2f} "
                  f"{g('lp_recall',ra):>5.2f} {g('lp_recall',rb):>5.2f} | "
                  f"{g('zs_iou',ra):>5.3f} {g('zs_iou',rb):>5.3f} "
                  f"{g('naive_iou',ra):>5.3f} {g('naive_iou',rb):>5.3f} "
                  f"{g('oracle_iou',ra):>5.3f} {g('oracle_iou',rb):>5.3f} "
                  f"{g('oracle_gain',ra):>5.2f} {g('oracle_gain',rb):>5.2f}")

    # ---- Q1 / Q2 / Q3 summary ----
    print(f"\n{'='*80}\n=== Q1: what made crosstalk TTA better (per-class naive-gain) ===\n{'='*80}")
    for cond in ['crosstalk']:
        classes = sorted(data[la][cond]['per_class'].keys(), key=int)
        print(f"{'cls':>3} {'naive_gain_A':>12} {'naive_gain_B':>12} {'d(feat_cos)':>12} {'d(lp_recall)':>12}")
        for c in classes:
            ra, rb = data[la][cond]['per_class'][c], data[lb][cond]['per_class'][c]
            if ra['naive_iou'] != ra['naive_iou'] or rb['naive_iou'] != rb['naive_iou']:
                continue
            print(f"{int(c):>3} {ra['naive_iou']-ra['zs_iou']:>12.3f} {rb['naive_iou']-rb['zs_iou']:>12.3f} "
                  f"{rb['feat_cos']-ra['feat_cos']:>12.2f} {rb['lp_recall']-ra['lp_recall']:>12.2f}")

    print(f"\n{'='*80}\n=== Q2: what made fog worse (per-class oracle) ===\n{'='*80}")
    classes = sorted(data[la]['fog']['per_class'].keys(), key=int)
    print(f"{'cls':>3} {'oracle_A':>8} {'oracle_B':>8} {'d(oracle)':>10} {'d(feat_cos)':>12} {'d(dir_ret)':>12}")
    for c in classes:
        ra, rb = data[la]['fog']['per_class'][c], data[lb]['fog']['per_class'][c]
        print(f"{int(c):>3} {ra['oracle_iou']:>8.3f} {rb['oracle_iou']:>8.3f} "
              f"{rb['oracle_iou']-ra['oracle_iou']:>10.3f} {rb['feat_cos']-ra['feat_cos']:>12.2f} "
              f"{rb['dir_retention']-ra['dir_retention']:>12.2f}")

    print(f"\n{'='*80}\n=== Q3: does clean-anchoring cap the fog ceiling? ===\n{'='*80}")
    for cond in CONDS:
        for label in (la, lb):
            pc = data[label][cond]['per_class']
            pairs = [(pc[str(c)]['feat_cos'], pc[str(c)]['oracle_gain'],
                      pc[str(c)]['dir_retention'], pc[str(c)]['oracle_iou'])
                     for c in pc
                     if pc[str(c)]['feat_cos'] == pc[str(c)]['feat_cos']
                     and pc[str(c)]['oracle_gain'] == pc[str(c)]['oracle_gain']]
            fcos = [p[0] for p in pairs]; ogain = [p[1] for p in pairs]
            dire = [p[2] for p in pairs]; orc = [p[3] for p in pairs]
            print(f"{label:<12} {cond:<10} rho(feat_cos, oracle_gain)={spearman(fcos, ogain):+.2f}   "
                  f"rho(dir_retention, oracle)={spearman(dire, orc):+.2f}")

    # ---- active-learning readiness comparison ----
    print(f"\n{'='*80}\n=== Active-learning readiness: one label per class-mean cluster ===\n{'='*80}")
    for cond in CONDS:
        a_agg, b_agg = data[la][cond]['aggregate'], data[lb][cond]['aggregate']
        print(f"{cond}: one-label-per-cluster purity  {la} {a_agg.get('al_purity', float('nan')):.3f}  |  "
              f"{lb} {b_agg.get('al_purity', float('nan')):.3f}")
        print(f"{'cls':>3} {'alP_A':>6} {'alP_B':>6} | {'orA':>5} {'orB':>5} | {'tightA':>6} {'tightB':>6}")
        for c in sorted(data[la][cond]['per_class'].keys(), key=int):
            ra, rb = data[la][cond]['per_class'][c], data[lb][cond]['per_class'][c]
            g = lambda k, d: float('nan') if d.get(k) is None or d[k] != d[k] else d[k]
            print(f"{int(c):>3} {g('al_purity', ra):>6.2f} {g('al_purity', rb):>6.2f} | "
                  f"{g('oracle_iou', ra):>5.2f} {g('oracle_iou', rb):>5.2f} | "
                  f"{g('corr_tightness', ra):>6.2f} {g('corr_tightness', rb):>6.2f}")

    # ---- two-branch decoupling gate (only meaningful when --inv_ch is set) ----
    if args.inv_ch and 'inv_feat_cos' in data[lb][CONDS[0]]['per_class'].get(str(sorted(
            data[lb][CONDS[0]]['per_class'].keys(), key=int)[0]), {}):
        print(f"\n{'='*80}\n=== DECOUPLING GATE (--inv_ch {args.inv_ch}): per-branch structure ===\n{'='*80}")
        print(f"Goal: corr dir_retention < 1 (shift retained in the corr branch) while the inv "
              f"branch stays anchored; concatenated oracle up. Report: {la} -> {lb}.")
        for cond in CONDS:
            pc = data[lb][cond]['per_class']
            agg = data[lb][cond]['aggregate']
            print(f"\n{cond}: {lb} aggregate zs {agg['zs']:.4f} naive {agg['naive']:.4f} "
                  f"oracle {agg['oracle']:.4f}")
            print(f"{'cls':>3} {'inv_fc':>7} {'inv_dir':>7} {'corr_fc':>7} {'corr_dir':>7} "
                  f"{'corr_t':>7} {'oracle':>7} {'or_gain':>7}")
            for c in sorted(pc.keys(), key=int):
                r = pc[c]
                g = lambda k, d: float('nan') if d.get(k) is None or d[k] != d[k] else d[k]
                print(f"{int(c):>3} {g('inv_feat_cos', r):>7.2f} {g('inv_dir_retention', r):>7.2f} "
                      f"{g('corr_feat_cos', r):>7.2f} {g('corr_dir_retention', r):>7.2f} "
                      f"{g('corr_tightness', r):>7.2f} {g('oracle_iou', r):>7.3f} "
                      f"{g('oracle_gain', r):>7.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(data, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
