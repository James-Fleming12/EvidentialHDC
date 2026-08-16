"""hdc_rule_bench.py: efficiency benchmark of the R1 (distance-to-prototype) vs
R4 (linear probe on the HDC code) decoders (eval-only, quick).

The C10 diagnostic validated R4 on mIoU (ceiling 1.2-1.8x over R1). This measures
the EFFICIENCY side the user asked to check: fit cost, decode cost, and memory for
the two decision rules on the same frozen cov-shift features.

  R1 : build_hdc_prototypes (per-class mean of the sign codes) + cosine argmax.
  R4 : sklearn LogisticRegression fit on the 10k-d code + .predict.

Measured on the CLEAN pool (the fit/zero-shot cost) and one corrupted condition
(the pool-refit oracle cost), on ep10 + ep21 cov-shift weights.

Usage:
  uv run python robust_diagnostic/hdc_rule_bench.py \
    --path_b <ckpt> --method_b supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --out robust_diagnostic/logs/hdc_rule_bench_covshift_ep10.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import resource
import yaml
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

NUM_CLASSES = 17
DGLSSPP_PATH = 'robust_diagnostic/logs/supcon_vib_dglsspp'
DGLSSPP_METHOD = 'supcon_vib_dglsspp'


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


def hdc_codes(feats, proj, device, chunk=200000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)


def bench(label, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
    return out, dt, rss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--max_clean", type=int, default=200000)
    parser.add_argument("--cond", type=str, default="wet_ground")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/hdc_rule_bench.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)

    # bounded clean sample for the probe fit / spreads (R4 only needs a subsample)
    max_clean = min(args.max_clean, len(fa))
    ci = torch.randperm(len(fa))[:max_clean]
    fa_s, la_s = fa[ci], la[ci]

    print(f"\n{'='*90}")
    print(f"=== {args.label}: R1 (prototype) vs R4 (linear probe) efficiency ===")
    print(f"{'='*90}")

    results = {}

    # --- fit cost (clean) ---
    print("\n[clean fit]")
    # R1: prototypes from the full clean pool
    _, t_r1_fit, rss1 = bench("R1-fit", lambda: build_hdc_prototypes(fa, la, proj, device=device))
    # R4: probe fit on the bounded clean codes
    clean_codes = hdc_codes(fa_s, proj, device)
    def r4_fit():
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(clean_codes[:100000].numpy(), la_s[:100000].numpy())
        return clf
    clf_hdc, t_r4_fit, rss4 = bench("R4-fit", r4_fit)
    print(f"  R1 build_hdc_prototypes: {t_r1_fit:.2f}s   (full {len(fa)} clean points)")
    print(f"  R4 LogisticRegression:   {t_r4_fit:.2f}s   (on {min(100000, len(clean_codes))} clean codes, 10k-d)")

    # --- decode cost on the corrupted val (the per-frame deployment cost) ---
    print(f"\n[decode: {args.cond} val, {args.val_size} points]")
    cdir = os.path.join(args.kittic_dir, args.cond, 'heavy')
    if not os.path.exists(cdir):
        cdir = os.path.join(args.kittic_dir, args.cond, 'moderate')
    f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
    torch.manual_seed(42)
    perm = torch.randperm(len(f))
    val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]
    codes = hdc_codes(val, proj, device)

    base_protos, base_lbls = build_hdc_prototypes(fa, la, proj, device=device)
    _, t_r1_dec, _ = bench("R1-decode", lambda: (
        F.normalize(codes.to(device), p=2, dim=1) @ base_protos.T).argmax(dim=1).cpu())
    _, t_r4_dec, _ = bench("R4-decode", lambda: torch.tensor(clf_hdc.predict(codes[:len(val)].numpy())))
    print(f"  R1 cosine argmax:  {t_r1_dec:.2f}s")
    print(f"  R4 predict:        {t_r4_dec:.2f}s")

    # --- pool-refit oracle cost (the per-condition adaptation cost) ---
    print(f"\n[pool-refit oracle: {args.cond}, {args.pool_size} pool points]")
    pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
    pool_codes = hdc_codes(pool, proj, device)
    _, t_r1_or, _ = bench("R1-oracle", lambda: build_hdc_prototypes(pool, pl, proj, device=device))
    def r4_or():
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(pool_codes[:100000].numpy(), pl[:100000].numpy())
        return clf
    _, t_r4_or, _ = bench("R4-oracle", r4_or)
    print(f"  R1 pool prototype re-estimate: {t_r1_or:.2f}s")
    print(f"  R4 pool probe re-fit:          {t_r4_or:.2f}s")

    # mIoU sanity (confirm the efficiency comparison is on the SAME quality result)
    r1_miou = compute_miou((F.normalize(codes.to(device), p=2, dim=1) @ base_protos.T).argmax(dim=1).cpu(), vl)
    r4_miou = compute_miou(torch.tensor(clf_hdc.predict(codes.numpy())), vl)
    print(f"\n  mIoU check on {args.cond}: R1 {r1_miou:.4f}  R4 {r4_miou:.4f}")

    results = {
        'label': args.label, 'cond': args.cond,
        'clean_fit_s': {'r1': t_r1_fit, 'r4': t_r4_fit},
        'decode_s_per_val': {'r1': t_r1_dec, 'r4': t_r4_dec,
                             'n_val': int(len(val))},
        'pool_refit_s': {'r1': t_r1_or, 'r4': t_r4_or, 'n_pool': int(len(pool))},
        'peak_rss_mb': {'r1': rss1, 'r4': rss4},
        'miou': {'r1': r1_miou, 'r4': r4_miou},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
