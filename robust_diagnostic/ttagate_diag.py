"""ttagate_diag.py: the Iteration-2 gate, as a prototype-update weight, for each
extractor (~40-60 min, eval-only).

Iteration 2 found strong correct-vs-wrong signals: local density for supcon_vib
(AUROC 0.91) and feature norm for the DGLSS / DGLSS++ extractors (0.84-0.87).
This runs ONLY the relevant gate as a weighted-prototype-update weight per
extractor, on fog and crosstalk, and reports the mIoU and the fraction of the
oracle gap it closes. The other gates (naive, confidence, distance, BN, kNN) are
already reported in the doc (Iteration 1); zero-shot and oracle are recomputed
here as the same-split references for the gap-closed fraction.

Per-extractor gate: supcon_vib -> density, supcon_vib_dglss / dglsspp -> norm.

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
GATES = {'supcon_vib': 'dens_gate', 'supcon_vib_dglss': 'norm_gate',
         'supcon_vib_dglsspp': 'norm_gate', 'supcon_vib_dglsspp_corsupcon': 'norm_gate'}
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
    parser.add_argument("--methods", type=str, default=",".join(GATES),
                        help="comma-separated subset of the extractors to evaluate")
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--med", action="store_true",
                        help="use medium-scale checkpoints (logs/med_pretrain_supcon_vib for "
                             "supcon_vib, the current medium DGLSS++ run) instead of the micro ones")
    parser.add_argument("--path", type=str, default="",
                        help="single checkpoint dir to evaluate (overrides --methods/--med)")
    parser.add_argument("--method", type=str, default="supcon_vib_dglsspp_corsupcon")
    parser.add_argument("--label", type=str, default="robust_21ep")
    parser.add_argument("--out", type=str, default=None,
                        help="output JSON (default: robust_diagnostic/logs/ttagate_results[_med].json)")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    out = args.out or os.path.join(args.log_dir, 'ttagate_results'
                                   + (('_' + args.label) if args.path
                                      else ('_med' if args.med else '')) + '.json')
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    results = {}
    if args.path:
        sel = [args.method]
        overrides = {args.method: args.path}
    else:
        sel = [m.strip() for m in args.methods.split(',') if m.strip()]
        overrides = {}

    for method in sel:
        if method not in GATES:
            continue
        gate = GATES[method]
        log_dir = (overrides.get(method)
                   or (MED_PATHS.get(method, os.path.join(args.log_dir, method))
                       if args.med else os.path.join(args.log_dir, method)))
        print(f"\n{'='*80}\n=== {method} ({log_dir}): {gate} as update weight ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=log_dir, method=method)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(clean_f[:min(100000, len(clean_f))].numpy(),
                clean_l[:min(100000, len(clean_l))].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)

        print(f"{'cond':<10} {'zs':>7} {'oracle':>8} {gate:>9}   gap-closed")
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
            ones = torch.ones(len(pool), device=device)

            if gate == 'dens_gate':
                # dens = -mean top-20 cosine (higher = sparser = more likely correct).
                # A raw clamp(min=0) zeroes out nearly every weight (dens is almost
                # always negative in clustered high-dim space), so the update becomes
                # a no-op. Shift instead: monotone increasing, all positive.
                dens = local_density(pool)
                w = (dens - dens.min()) + 1e-3
            else:
                w = pool.norm(p=2, dim=1).to(device)

            def decode(protos):
                return proto_miou(val, vl, protos, proto_lbls, proj, device)

            zs = decode(base_protos)
            oracle = decode(weighted_mean_update(base_protos, proto_lbls, pool, l[pool_idx].to(device),
                                                 ones, proj, device))
            g = decode(weighted_mean_update(base_protos, proto_lbls, pool, lp_preds, w, proj, device))
            gap = (g - zs) / (oracle - zs) if oracle > zs else float('nan')
            r_cond[cond] = {'zero_shot': zs, 'oracle': oracle, gate: g, 'gap_closed': gap}
            print(f"{cond:<10} {zs:>7.4f} {oracle:>8.4f} {g:>9.4f}   {gap:.2f}")
        results[method] = r_cond

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
