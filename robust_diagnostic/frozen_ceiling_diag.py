"""frozen_ceiling_diag.py: labeled ceiling of the three frozen feature extractors (~45 min, eval-only).

For each trained feature extractor (supcon_vib, supcon_vib_dglss, supcon_vib_dglsspp),
report per condition what a LABELED decoder can achieve on the frozen 128D features:
  - linear probe accuracy and mIoU (the continuous-space labeled ceiling; the probe
    is fit on clean labels and evaluated frozen),
  - the full-label HDC oracle mIoU (prototypes re-estimated on the corrupted points
    with true labels, then decoded; the binarized labeled ceiling),
  - the frozen zero-shot HDC mIoU (reference).

This is the "ceiling" comparison across the three extractors: which frozen features
give the highest recoverable performance per condition, independent of any TTA.

Usage:
  uv run python robust_diagnostic/frozen_ceiling_diag.py
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

CONDS = ['fog', 'crosstalk', 'snow', 'wet_ground', 'incomplete_echo',
         'beam_missing', 'motion_blur', 'cross_sensor']
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
    parser.add_argument("--kitti_dir", type=str, default="/mnt/alpha/jmfleming/KITTI")
    parser.add_argument("--kittic_dir", type=str, default="/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C")
    parser.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
    parser.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
    parser.add_argument("--log_dir", type=str, default="robust_diagnostic/logs")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--med", action="store_true",
                        help="use medium-scale checkpoints (logs/med_pretrain_supcon_vib for "
                             "supcon_vib, the current medium DGLSS++ run) instead of the micro ones")
    parser.add_argument("--method", type=str, default="supcon_vib_dglsspp_corsupcon",
                        help="GenTrainer method name (used with --path)")
    parser.add_argument("--path", type=str, default="",
                        help="single checkpoint dir to evaluate (overrides the default method loop)")
    parser.add_argument("--label", type=str, default="robust_21ep",
                        help="label for the single --path checkpoint")
    parser.add_argument("--out", type=str, default=None,
                        help="output JSON (default: robust_diagnostic/logs/frozen_ceiling_results[_med].json)")
    args = parser.parse_args()

    DATA = yaml.safe_load(open(args.config, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    if args.path:
        targets = [(args.method, args.path, args.label)]
    else:
        targets = [(m, (MED_PATHS.get(m, os.path.join(args.log_dir, m))
                        if args.med else os.path.join(args.log_dir, m)), m) for m in METHODS]
    out = args.out or os.path.join(args.log_dir, 'frozen_ceiling_results'
                                   + (('_' + args.label) if args.path
                                      else ('_med' if args.med else '')) + '.json')
    proj = get_hdc_projection(dim_in=128, dim_out=10000, device=device)
    results = {}

    for method, log_dir, label in targets:
        print(f"\n{'='*80}\n=== {method} {label} ({log_dir}): frozen, labeled ceiling ===\n{'='*80}")
        trainer = GenTrainer(ARCH, DATA, args.kitti_dir, log_dir, path=log_dir, method=method)
        model = trainer.model

        clean_parser = build_parser(args.kitti_dir, DATA, ARCH)
        clean_f, clean_l = extract_features(model, clean_parser, device, args.frames)

        clf = LogisticRegression(max_iter=1000)
        fit_n = min(100000, len(clean_f))
        clf.fit(clean_f[:fit_n].numpy(), clean_l[:fit_n].numpy())
        base_protos, proto_lbls = build_hdc_prototypes(clean_f, clean_l, proj, device=device)

        r_m = {}
        print(f"{'cond':<16} {'LP-acc':>7} {'LP-mIoU':>8} {'HDC-zs':>8} {'HDC-oracle':>10}")
        for cond in CONDS:
            cdir = os.path.join(args.kittic_dir, cond, 'heavy')
            if not os.path.exists(cdir):
                cdir = os.path.join(args.kittic_dir, cond, 'moderate')
            f, l = extract_features(model, build_parser(cdir, DATA, ARCH), device, args.frames)
            n = min(200000, len(f))
            lp_preds = clf.predict(f[:n].cpu().numpy())
            lp_acc = float((lp_preds == l[:n].numpy()).mean())
            lp_miou = compute_miou(torch.tensor(lp_preds), l[:n])
            hdc_zs = proto_miou(f, l, base_protos, proto_lbls, proj, device)
            # full-label HDC oracle: re-estimate prototypes on the corrupted points
            oracle_protos, oracle_lbls = build_hdc_prototypes(f, l, proj, device=device)
            hdc_oracle = proto_miou(f, l, oracle_protos, oracle_lbls, proj, device)
            r_m[cond] = {'lp_acc': lp_acc, 'lp_miou': lp_miou,
                         'hdc_zs': hdc_zs, 'hdc_oracle': hdc_oracle}
            print(f"{cond:<16} {lp_acc:>7.4f} {lp_miou:>8.4f} {hdc_zs:>8.4f} {hdc_oracle:>10.4f}")
        results[label] = r_m

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
