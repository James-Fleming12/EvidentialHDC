"""hdc_rule_diag.py: the C8 decision-rule diagnostic (eval-only).

C8 proved the cov-shift healthy-ceiling loss survives every ENCODING change
(sign/bias/zscore/fourier). This diagnostic tests the OTHER decoder-side lever: the
DECISION RULE. All C7/C8 variants ended in the same nearest-centroid rule (unit-norm
cosine to per-class prototypes). The C6 packing loss is per-class, so a
class-conditional rule is the untested alternative.

For each frozen feature extractor (default: the ep10/ep21 cov-shift weights) on the
healthy conditions + fog/crosstalk, reports zero-shot and oracle mIoU under three
decision rules:

  R1 baseline : unit-norm cosine to per-class prototypes (the current rule).
  R2 scaled   : per-class scaled cosine -- each prototype's similarity is divided by
                that class's within-class spread (1/corr_tight proxy), re-tightening
                the per-class decision boundary. Method-preserving: keeps HDC codes +
                prototypes + the label-free re-estimation machinery.
  R3 probe    : learned 128-d LogisticRegression decision rule (fit on clean features,
                evaluated frozen) -- the continuous-space labeled ceiling.
  R4 hdc-probe: the headline C8 question restated -- is the HDC CODE (the binarized
                10k-d space) itself linearly separable? A LogisticRegression fit on the
                sign code, clean-fit (zs) and pool-refit (oracle). If R4 >> R1, the
                nearest-centroid rule throws away a strong linear signal in the code
                that a learned decision rule on the HDC code would capture.

The decision is: does R2 recover the healthy-condition oracle (B) toward R1's DGLSS++
level WITHOUT losing the fog/crosstalk gain? R3 is the strong reference (what a
learned continuous rule achieves).

Usage:
  uv run python robust_diagnostic/hdc_rule_diag.py \
    --path_b robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan \
    --method_b supcon_vib_dglsspp_inputin_in_chan --label_b covshift_ep10 \
    --conds snow,wet_ground,fog,crosstalk \
    --out robust_diagnostic/logs/hdc_rule_covshift_ep10.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import yaml
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

CONDS_DEFAULT = ['snow', 'wet_ground', 'fog', 'crosstalk']
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

def hdc_codes(feats, proj, device, chunk=100000):
    codes = []
    for s in range(0, len(feats), chunk):
        codes.append(torch.sign(feats[s:s + chunk].to(device) @ proj).cpu())
    return torch.cat(codes, dim=0)

def per_class_spread(codes, lbls, proto_lbls):
    """Per-class within-class spread in the binarized code (1 = the corr_tight proxy):
    mean cosine of a class's points to its own prototype. A class whose points spread
    (low corr_tight) gets a DOWN-weighted similarity in R2."""
    spread = {}
    for c in proto_lbls.tolist():
        m = lbls == c
        if int(m.sum().item()) < 500:
            spread[c] = 1.0
            continue
        cc = codes[m][:20000]
        proto = F.normalize(cc.mean(dim=0, keepdim=True), p=2, dim=1)
        spread[c] = float((F.normalize(cc, p=2, dim=1) @ proto.T).mean().item())
    return spread

def decode_rule1(codes, protos, proto_lbls, device):
    """R1: unit-norm cosine to prototypes (the current rule)."""
    preds = []
    for s in range(0, len(codes), 100000):
        sims = F.normalize(codes[s:s + 100000].to(device), p=2, dim=1) @ protos.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

def decode_rule2(codes, protos, proto_lbls, spread, device):
    """R2: per-class scaled cosine. Each class's similarity is divided by its spread
    (inverse corr_tight), re-tightening the decision boundary for classes whose points
    have spread under the corruption."""
    scale = torch.tensor([spread.get(c, 1.0) for c in proto_lbls.tolist()], device=device)
    protos_w = protos / scale.unsqueeze(1).clamp(min=0.05)
    preds = []
    for s in range(0, len(codes), 100000):
        sims = F.normalize(codes[s:s + 100000].to(device), p=2, dim=1) @ protos_w.T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    return torch.cat(preds)

def decode_rule3(feats, clf):
    """R3: learned 128-d LogisticRegression decision rule (fit on clean, frozen)."""
    n = len(feats)
    preds = torch.tensor(clf.predict(feats[:min(n, 500000)].numpy()))
    return preds

def decode_rule4(codes, clf_hdc):
    """R4: linear probe fit on the HDC CODE (the binarized 10k-d space) itself.
    The headline C8 question restated: is the HDC space linearly separable in a way
    the nearest-centroid (cosine-to-prototype) rule MISSES? Fit a LogisticRegression on
    the 10k-d sign code (clean-fit = frozen reference, or pool-refit = oracle). If R4
    >> R1, the current implementation throws away a strong linear signal in the code."""
    n = len(codes)
    preds = torch.tensor(clf_hdc.predict(codes[:min(n, 500000)].numpy()))
    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--max_clean", type=int, default=200000,
                        help="cap on the clean points used for the R4 HDC-code probe fit "
                             "(binarizing the full 8M-point clean pool is 320GB; the probe "
                             "only needs a bounded subsample)")
    parser.add_argument("--path_b", type=str, default=DGLSSPP_PATH)
    parser.add_argument("--method_b", type=str, default=DGLSSPP_METHOD)
    parser.add_argument("--label_b", type=str, default="covshift")
    parser.add_argument("--conds", type=str, default=",".join(CONDS_DEFAULT))
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/hdc_rule_results.json")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]

    trainer = GenTrainer(ARCH, DATA, args.kitti_dir, args.path_b, path=args.path_b,
                         method=args.method_b)
    model = trainer.model

    clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
    fa, la = extract_features(model, clean_parser, device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    base_protos, base_lbls = build_hdc_prototypes(fa, la, proj, device=device)

    clf = LogisticRegression(max_iter=1000)
    fit_n = min(100000, len(fa))
    clf.fit(fa[:fit_n].numpy(), la[:fit_n].numpy())

    # R4: is the HDC CODE itself linearly separable? Fit a LogisticRegression on the
    # binarized 10k-d code. clean-fit = frozen reference (zs); per-condition pool-refit
    # = oracle. If R4 >> R1, the nearest-centroid rule misses a strong linear signal.
    # BOUNDED: binarizing the full 8M-point clean pool is 8M x 10000 = 320GB. The R4
    # probe fit only needs a capped subsample (and it's a probe, not a prototype
    # estimate), so compute the clean codes once on a bounded random sample.
    max_clean = min(args.max_clean, len(fa))
    clean_idx = torch.randperm(len(fa))[:max_clean]
    fa_s, la_s = fa[clean_idx], la[clean_idx]
    clean_codes = hdc_codes(fa_s, proj, device)
    clf_hdc = LogisticRegression(max_iter=1000, C=1.0)
    hdc_fit_n = min(100000, len(clean_codes))
    clf_hdc.fit(clean_codes[:hdc_fit_n].numpy(), la_s[:hdc_fit_n].numpy())
    # clean spreads for R2-zs also use the bounded sample
    spread_clean = per_class_spread(clean_codes, la_s, base_lbls)

    results = {}
    print(f"\n{'='*100}\n=== {args.label_b}: HDC decision-rule diagnostic ===\n{'='*100}")
    print(f"{'cond':<14} {'R1-zs':>7} {'R2-zs':>7} {'R3-zs':>7} {'R4-zs':>7} | {'R1-orc':>7} {'R2-orc':>7} {'R3-orc':>7} {'R4-orc':>7}")
    for cond in conds:
        cdir = os.path.join(args.kittic_dir, cond, 'heavy')
        if not os.path.exists(cdir):
            cdir = os.path.join(args.kittic_dir, cond, 'moderate')
        f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
        torch.manual_seed(42)
        perm = torch.randperm(len(f))
        pool, pl = f[perm[:args.pool_size]], l[perm[:args.pool_size]]
        val, vl = f[perm[-args.val_size:]], l[perm[-args.val_size:]]

        codes = hdc_codes(val, proj, device)
        # zero-shot: frozen clean prototypes / probes, all four rules
        r1_zs = compute_miou(decode_rule1(codes, base_protos, base_lbls, device), vl)
        # R2 zero-shot uses the clean class spreads computed once above
        r2_zs = compute_miou(decode_rule2(codes, base_protos, base_lbls, spread_clean, device), vl)
        r3_zs = compute_miou(decode_rule3(val, clf), vl)
        r4_zs = compute_miou(decode_rule4(codes, clf_hdc), vl)

        # oracle: re-estimate prototypes from the corrupted labeled pool
        pool_codes = hdc_codes(pool, proj, device)
        orc_protos, orc_lbls = build_hdc_prototypes(pool, pl, proj, device=device)
        spread_pool = per_class_spread(pool_codes, pl, orc_lbls)
        r1_or = compute_miou(decode_rule1(codes, orc_protos, orc_lbls, device), vl)
        r2_or = compute_miou(decode_rule2(codes, orc_protos, orc_lbls, spread_pool, device), vl)
        r3_or = compute_miou(decode_rule3(val, clf), vl)  # probe is frozen; same as zs
        # R4 oracle: refit the HDC-code probe on the corrupted labeled pool
        pool_probe_n = min(100000, len(pool_codes))
        clf_hdc_pool = LogisticRegression(max_iter=1000, C=1.0)
        clf_hdc_pool.fit(pool_codes[:pool_probe_n].numpy(), pl[:pool_probe_n].numpy())
        r4_or = compute_miou(decode_rule4(codes, clf_hdc_pool), vl)

        results[cond] = {
            'r1_zs': r1_zs, 'r2_zs': r2_zs, 'r3_zs': r3_zs, 'r4_zs': r4_zs,
            'r1_oracle': r1_or, 'r2_oracle': r2_or, 'r3_oracle': r3_or,
            'r4_oracle': r4_or,
        }
        print(f"{cond:<14} {r1_zs:>7.4f} {r2_zs:>7.4f} {r3_zs:>7.4f} {r4_zs:>7.4f} | "
              f"{r1_or:>7.4f} {r2_or:>7.4f} {r3_or:>7.4f} {r4_or:>7.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {args.out}")
    print("\n=== WHAT TO LOOK FOR ===")
    print("On the healthy conditions (snow/wet_ground):")
    print("  - R1 (baseline cosine) oracle is the C8 ceiling loss (B ~0.22 ep10).")
    print("  - R2 (per-class scaled cosine) oracle: does it recover toward plain")
    print("    DGLSS++'s ~0.27 WITHOUT dropping the fog/crosstalk R1 oracle? That is the")
    print("    class-conditional decision-rule fix.")
    print("  - R3 (learned 128-d probe) is the continuous labeled ceiling reference.")
    print("  - R4 (linear probe on the HDC CODE) answers the headline C8 question: is the")
    print("    binarized space itself linearly separable in a way the nearest-centroid")
    print("    rule misses? If R4_oracle >> R1_oracle on the healthy conditions, the")
    print("    current implementation throws away a strong linear signal -> a learned")
    print("    decision rule on the HDC code is the fix.")
    print("If R2 or R4 recovers healthy but holds fog/crosstalk, adopt it decoder-side.")

if __name__ == "__main__":
    main()
