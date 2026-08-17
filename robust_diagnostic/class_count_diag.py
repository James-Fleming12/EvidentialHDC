"""class_count_diag.py: does the label granularity hurt robustness? (~5h run)

D3CTTA / GIPSO use a 7-class SemanticKITTI map (vehicle, pedestrian, road, sidewalk,
terrain, manmade, vegetation; see thirdparty/D3CTTA/utils/_resources/semantic-kitti.yaml),
folding the fragile rare classes (bicycle, truck, bus, motorcycle, traffic-sign) into
background or broad superclasses. Our default config is a 17-way head (~12 populated
classes) that keeps those rare classes separate.

Hypothesis to test: the fine granularity is part of why fog/crosstalk collapse for the
HDC pipeline. The rare classes absorb into neighbors under corruption and poison the
prototypes; under the coarse 7-class regime the same corruption is expressed on classes
with far more support, so the encoder and the HDC decode should survive better.

Design: train supcon_vib under BOTH configs at the SAME budget, then compare the
clean / fog / crosstalk headroom (linear probe + HDC prototype mIoU) and the
robustness gap (1 - fog/clean). The budget is equal across configs so the comparison
is controlled. Default (10 epochs at 50% data) is ~5h for the two runs plus evals.

Usage:
  uv run python robust_diagnostic/class_count_diag.py
  uv run python robust_diagnostic/class_count_diag.py --configs semantic-kitti-all,semantic-kitti-7
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
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

# (label, config path)
CONFIGS = [
    ('all17', 'config/labels/semantic-kitti-all.yaml'),
    ('seven', 'config/labels/semantic-kitti-7.yaml'),
]
CONDS = ['fog', 'crosstalk']

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

def proto_miou(feats, lbls, base_protos, proto_lbls, proj, device):
    feats_d = feats.to(device)
    protos = F.normalize(base_protos, p=2, dim=1)
    sims = []
    for start in range(0, len(feats_d), 50000):
        hc = F.normalize(torch.sign(feats_d[start:start + 50000] @ proj), p=2, dim=1)
        sims.append(hc @ protos.T)
    sims = torch.cat(sims, dim=0)
    return compute_miou(proto_lbls[sims.argmax(dim=1)], lbls.to(device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=",".join(c[0] for c in CONFIGS))
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--epochs", type=int, default=10, help="epochs per config (~5h total at --cutoff 0.5)")
    parser.add_argument("--cutoff", type=float, default=0.5, help="fraction of training data per epoch")
    parser.add_argument("--frames", type=int, default=50, help="frames per condition for evaluation")
    args = parser.parse_args()

    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    cfg_labels = [c.strip() for c in args.configs.split(',') if c.strip()]
    configs = [c for c in CONFIGS if c[0] in cfg_labels]
    results = {}

    for label, cfg_path in configs:
        DATA = yaml.safe_load(open(cfg_path, 'r'))
        n_cls = len(DATA["learning_map_inv"])
        print(f"\n{'='*80}\n=== {label}: {cfg_path} ({n_cls}-class head) ===\n{'='*80}")

        log_dir = os.path.join(args.log_dir, f"classcount_{label}")
        os.makedirs(log_dir, exist_ok=True)
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, method='supcon_vib',
                             cutoff_percent=args.cutoff)
        trainer.train(epochs=args.epochs)
        model = trainer.model

        proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        clean_f, clean_l = extract_features(model, clean_parser, device, args.frames)
        print(f"  clean n {len(clean_f)} (classes present: {sorted(set(clean_l.tolist()))})")

        clf = LogisticRegression(max_iter=1000)
        fit_n = min(100000, len(clean_f))
        clf.fit(clean_f[:fit_n].numpy(), clean_l[:fit_n].numpy())
        clean_lp = float(clf.score(clean_f[:fit_n].numpy(), clean_l[:fit_n].numpy()))
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        clean_pm = proto_miou(clean_f, clean_l, base_protos, proto_lbls, proj, device)

        cond_rows = {'clean_lp': clean_lp, 'clean_pm': clean_pm}
        print(f"{'cond':<12} {'LP':>7} {'HDC-proto':>9} {'gap':>6}")
        print(f"{'clean':<12} {clean_lp:>7.4f} {clean_pm:>9.4f} {'-':>6}")
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            lp = float(clf.score(f[:200000].cpu().numpy(), l[:200000].numpy()))
            pm = proto_miou(f, l, base_protos, proto_lbls, proj, device)
            gap_lp = 1.0 - lp / clean_lp if clean_lp > 0 else float('nan')
            gap_pm = 1.0 - pm / clean_pm if clean_pm > 0 else float('nan')
            print(f"{cond:<12} {lp:>7.4f} {pm:>9.4f} {gap_lp:>6.3f} (LP gap) / {gap_pm:.3f} (proto gap)")
            cond_rows[cond] = {'lp': lp, 'proto_miou': pm, 'lp_gap': gap_lp, 'proto_gap': gap_pm}
        results[label] = cond_rows

    print(f"\n{'='*80}\n=== Summary (robustness gap = 1 - corrupt/clean) ===")
    for label, r in results.items():
        for cond in CONDS:
            c = r.get(cond, {})
            print(f"{label:<8} {cond:<12} clean-pm {r['clean_pm']:.4f} | "
                  f"corrupt-pm {c.get('proto_miou', float('nan')):.4f} | "
                  f"proto-gap {c.get('proto_gap', float('nan')):.3f} | "
                  f"LP-gap {c.get('lp_gap', float('nan')):.3f}")

    out_path = os.path.join(args.log_dir, 'class_count_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
