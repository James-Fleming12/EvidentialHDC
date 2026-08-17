"""dglss_bug_probe.py: reproduce the old DGLSS flatline (~10 min, micro scale).

Question: was the old DGLSSTrainer (DensityUnsupHyperLidar) flatline an
implementation bug or inherent to DGLSS? The old code computed the LSCC/SCC
class-prototype correlations on RAW, UNNORMALIZED means, so the Gram entries scale
as ||z||^2 and the loss as ||z||^4, which blows up (or, via loss dominance,
flatlines) during VIB-free training. Our port fixed this by normalizing the
prototypes. The FIXED form is already trained and measured (supcon_vib_dglss in
the isotropy run: clean HDC 0.389, fog 0.056, crosstalk 0.099). This probe trains
only the OLD (raw) form at the same micro budget to check whether it reproduces
the flatline / NaN while the fixed form trained cleanly.

Usage:
  uv run python robust_diagnostic/dglss_bug_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn.functional as F

from dataset.kitti.parser import Parser
from modules.gen_trainers import GenTrainer
from modules.oracle_core import get_hdc_projection, build_hdc_prototypes, compute_miou

# the old (raw, unnormalized) SCC prototype form, i.e. the bug-repro arm
ARMS = {'raw_old': {'dglss_scc_norm': False}}
# reference: the FIXED form, already trained at the same budget in the isotropy run
FIXED_REF = {'clean_hdc_miou': 0.389, 'fog_hdc_miou': 0.056, 'crosstalk_hdc_miou': 0.099}

def build_parser(root, data, arch):
    return Parser(root=root, train_sequences=['08'], valid_sequences=['08'],
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=arch["dataset"]["sensor"], max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def extract_features(model, parser, device, num_frames=25):
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
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs/dglss_bug_probe")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--frames", type=int, default=25)
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    results = {}
    for name, kwargs in ARMS.items():
        log_dir = os.path.join(args.log_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\n{'='*80}\n=== DGLSS, {name} SCC prototype form ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, method='supcon_vib_dglss',
                             cutoff_percent=args.cutoff, **kwargs)
        trainer.train(epochs=args.epochs)
        model = trainer.model

        clean_f, clean_l = extract_features(model, build_parser(args.kitti_dir, DATA, ARCH),
                                            device, args.frames)
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)
        clean_miou = proto_miou(clean_f, clean_l, base_protos, proto_lbls, proj, device)

        res = {'clean_hdc_miou': clean_miou}
        for cond in ('fog', 'crosstalk'):
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            res[f'{cond}_hdc_miou'] = proto_miou(f, l, base_protos, proto_lbls, proj, device)
        results[name] = res
        print(f"  clean HDC mIoU {res['clean_hdc_miou']:.4f} | "
              f"fog {res['fog_hdc_miou']:.4f} | crosstalk {res['crosstalk_hdc_miou']:.4f}")

    print(f"\n{'='*80}\n=== Comparison (old raw form vs fixed form, same budget) ===")
    print(f"  {'arm':<10} {'clean':>7} {'fog':>7} {'crosstalk':>9}")
    print(f"  {'fixed(ref)':<10} {FIXED_REF['clean_hdc_miou']:>7.4f} {FIXED_REF['fog_hdc_miou']:>7.4f} {FIXED_REF['crosstalk_hdc_miou']:>9.4f}")
    for name, r in results.items():
        print(f"  {name:<10} {r['clean_hdc_miou']:>7.4f} {r['fog_hdc_miou']:>7.4f} {r['crosstalk_hdc_miou']:>9.4f}")

    out = os.path.join(args.log_dir, 'results.json')
    os.makedirs(args.log_dir, exist_ok=True)
    import json
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
