"""tta_ceiling_diag.py: do the previous supcon_vib TTA difficulties still hold on
the DGLSS / DGLSS++ extractors? (~1.5-2h, eval-only)

Iteration 1 of the robust-encoder comparison. The supcon_vib space showed that the
labeled ceiling is unreachable without explicit labels: a label-free signal can say
WHICH points are wrong (detection is solvable) but not WHAT class they belong to
(assignment is the wall), because the true class of a recoverable point is invisible
to every global signal. This diagnostic rechecks those exact claims on the DGLSS
and DGLSS++ feature extractors, alongside the TTA methods that were developed
against the ceiling.

Per extractor (supcon_vib, supcon_vib_dglss, supcon_vib_dglsspp) and condition
(fog, crosstalk, snow control), the frozen 128D features are scored by:

  - SPACE DIAGNOSTICS (the "no label-free path" checks, reused from the earlier
    TTA iterations):
      * rec@3         : fraction of zero-shot-wrong points whose TRUE class is in
                        the top-3 clean prototypes (feature-intrinsic recoverability;
                        random baseline ~0.19 for 16 classes)
      * cosT          : mean cosine of zs-wrong points to their true centroid
      * rank-of-true  : mean rank of the true class among clean prototypes for the
                        recoverable points (was 3.7-4.8 on supcon_vib)
      * LP-on-recover : linear-probe accuracy on the recoverable (zs-wrong) points
                        (was 5-8% on supcon_vib: no global classifier names them)
      * gated-oracle / gated-LP: the assignment gap, oracle-assigned vs
                        LP-assigned re-estimate (was near-equal on crosstalk:
                        detection is solvable, assignment is not)
  - TTA METHODS (the label-free recovery attempts):
      * naive_ema, conf_gate, dist_gate : weighted prototype re-estimates
      * bn_align  : per-dimension statistic alignment, then frozen decode
      * knn_reassign : the recoverability-gated kNN reassignment (best prior)

Zero-shot and oracle are NOT re-run as results: they are already in the background
(Iteration 0.3, frozen labeled ceiling). assignment_gap_diag computes them as a
byproduct, and the report uses them only to frame the gap-closed fraction.

Usage:
  uv run python robust_diagnostic/tta_ceiling_diag.py
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

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import (get_hdc_projection, build_hdc_prototypes,
                                 compute_miou, weighted_mean_update)
from modules.tta_diagnostics import assignment_gap_diag, iter7_knn_reassign
from robust_diagnostic.d3ctta_diag import feature_recoverability

CONDS = ['fog', 'crosstalk', 'snow']
METHODS = ['supcon_vib', 'supcon_vib_dglss', 'supcon_vib_dglsspp']
# Medium-scale checkpoints used with --med (instead of the micro ones at log_dir/<method>).
# supcon_vib: the medium pretrain; supcon_vib_dglsspp: the current medium DGLSS++ run's
# output (the in-place isotropy_diag checkpoint). supcon_vib_dglss has no medium run yet.
MED_PATHS = {
    'supcon_vib': 'logs/med_pretrain_supcon_vib',
    'supcon_vib_dglsspp': 'robust_diagnostic/logs/supcon_vib_dglsspp',
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=500000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--med", action="store_true",
                        help="use medium-scale checkpoints (logs/med_pretrain_supcon_vib for "
                             "supcon_vib, the current medium DGLSS++ run) instead of the micro ones")
    parser.add_argument("--method", type=str, default="supcon_vib_dglsspp",
                        help="GenTrainer method name (used with --path)")
    parser.add_argument("--path", type=str, default="",
                        help="single checkpoint dir to evaluate (overrides the default method loop)")
    parser.add_argument("--label", type=str, default="single",
                        help="label for the single --path checkpoint")
    parser.add_argument("--out", type=str, default=None,
                        help="output JSON (default: robust_diagnostic/logs/tta_ceiling_results[_med].json)")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    if args.path:
        targets = [(args.label, args.method, args.path)]
    else:
        targets = [(m, m, (MED_PATHS.get(m, os.path.join(args.log_dir, m))
                           if args.med else os.path.join(args.log_dir, m)))
                   for m in METHODS]
    out = args.out or os.path.join(args.log_dir, 'tta_ceiling_results'
                                   + (('_' + args.label) if args.path else ('_med' if args.med else ''))
                                   + '.json')
    results = {}

    for label, method, log_dir in targets:
        print(f"\n{'='*80}\n=== {method} {label} ({log_dir}): ceiling-access diagnostics + TTA methods ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=log_dir, method=method)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        proj = get_hdc_projection(dim_in=clean_f.shape[1], dim_out=10000, device=device)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        clean_means = class_centroids(clean_f, clean_l)
        clean_stats = (clean_f.mean(0), clean_f.std(0) + 1e-6)

        header = (f"{'cond':<12} {'rec3':>6} {'cosT':>6} {'rankT':>6} {'LPrec':>6} "
                  f"{'gorc':>6} {'glp':>6} | {'naive':>7} {'conf':>7} {'dist':>7} "
                  f"{'bn':>7} {'knn':>7}  knnGap")
        print(header)
        r_cond = {}
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)

            # --- space diagnostics: the "no label-free path" claims ---
            ag = assignment_gap_diag(base_protos, proto_lbls, f, l, clf, proj, device,
                                     pool_size=args.pool_size, val_size=args.val_size)
            fr = feature_recoverability(f, l, clean_means)
            zs = ag['metrics']['zero_shot']
            oracle = ag['metrics']['oracle']
            rk = ag['rank_of_true_class']
            lp_rec = ag['lp_on']['recovered']
            gated = ag['gated']
            r_g = 0.25 if 0.25 in gated else next(iter(gated))
            g_orc = gated[r_g]['oracle_assigned']
            g_lp = gated[r_g]['lp_assigned']

            # --- TTA methods on the frozen features (pool/val split) ---
            torch.manual_seed(42)
            perm = torch.randperm(len(f))
            pool_idx, val_idx = perm[:args.pool_size], perm[-args.val_size:]
            pool = f[pool_idx]
            val, vl = f[val_idx], l[val_idx]

            lp_preds = torch.tensor(clf.predict(pool.numpy())).to(device)
            lp_conf = torch.tensor(clf.predict_proba(pool.numpy()).max(axis=1)).to(device)
            ones = torch.ones(len(pool), device=device)

            def decode(protos):
                return proto_miou(val, vl, protos, proto_lbls, proj, device)

            naive = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, ones, proj, device))
            conf_gate = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, lp_conf, proj, device))

            cm = F.normalize(torch.stack([clean_means[c] for c in sorted(clean_means)]), p=2, dim=1)
            dist_w = (F.normalize(pool, p=2, dim=1) @ cm.T).max(dim=1).values.clamp(min=0)
            dist_gate = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, dist_w, proj, device))

            zs_m = val.mean(0).to(device); zs_s = val.std(0).to(device) + 1e-6
            val_aligned = (val.to(device) - zs_m) / zs_s * clean_stats[1].to(device) + clean_stats[0].to(device)
            bn_align = proto_miou(val_aligned, vl, base_protos, proto_lbls, proj, device)

            i7 = iter7_knn_reassign(base_protos, proto_lbls, f, l, clf, proj, device,
                                    pool_size=args.pool_size, val_size=args.val_size)
            knn = i7['metrics']['zs_reestimate']

            row = {'rec3': fr['rec_of_wrong'], 'cosT': fr['true_cos'],
                   'rank_true': rk['recovered']['mean'],
                   'rank_true_frac3': rk['recovered'].get('frac_rank3', float('nan')),
                   'lp_on_recovered': lp_rec,
                   'gated_oracle': g_orc, 'gated_lp': g_lp,
                   'naive_ema': naive, 'conf_gate': conf_gate, 'dist_gate': dist_gate,
                   'bn_align': bn_align, 'knn': knn,
                   'zero_shot': zs, 'oracle': oracle}
            r_cond[cond] = row
            knn_gap = (knn - zs) / (oracle - zs) if oracle > zs else float('nan')
            print(f"{cond:<12} {fr['rec_of_wrong']:>6.3f} {fr['true_cos']:>6.3f} "
                  f"{rk['recovered']['mean']:>6.2f} {lp_rec:>6.3f} {g_orc:>6.3f} "
                  f"{g_lp:>6.3f} | {naive:>7.4f} {conf_gate:>7.4f} "
                  f"{dist_gate:>7.4f} {bn_align:>7.4f} {knn:>7.4f}  {knn_gap:.2f}")
        results[label] = r_cond

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
