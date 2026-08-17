"""nusc_cross_domain_diag.py: zero-shot + oracle of KITTI-trained cov-shift weights on NuScenes.

The cov-shift extractor was trained on SemanticKITTI. This diagnostic evaluates it on
real NuScenes (converted to KITTI format at /mnt/alpha/jmfleming/nuscenes_kitti,
32-beam sensor) to answer: is there a zero-shot -> oracle gap for TTA to take on the
CROSS-DATASET domain that did not exist on KITTI?

Both datasets share the 17-class learned space (0=ignore, 1-16 shared classes; the
parsers' learning_map does the mapping at load). The extractor z8 (128-d) is therefore
directly comparable across datasets.

Metrics (mirrors frozen_ceiling_diag, on NuScenes):
  - hdc_zs      : zero-shot HDC oracle = prototypes built from KITTI CLEAN features
                  (frozen), decode NuScenes.
  - hdc_oracle  : re-estimate prototypes from a NuScenes labeled pool, decode NuScenes.
  - lp_acc/miou : learned 128-d LogisticRegression fit on KITTI clean, evaluated frozen
                  on NuScenes (the continuous labeled ceiling).

The TTA-relevant quantity is gap = oracle - zs. A large gap = the extractor's frozen
prototypes are far from the NuScenes domain but the features still carry recoverable
structure (TTA can take it). A near-zero gap = the extractor transferred (or the
features lost recoverable structure). Compare against the SAME extractor's KITTI gap
(measured on KITTI clean/corruptions in the other diagnostics).

Sensor note (from unsup_kitti-nuscenes.py): NuScenes has 32 beams (vs KITTI 64) and
~1000 points/revolution. The projection is height=32, width=1024, fov_up=10,
fov_down=-30 so the CNN sees dense, continuous input.

Usage:
  uv run python robust_diagnostic/nusc_cross_domain_diag.py \
    --path robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan \
    --method supcon_vib_dglsspp_inputin_in_chan --label covshift_ep10 \
    --out robust_diagnostic/logs/nusc_covshift_ep10.json
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

KITTI_DIR = '/mnt/alpha/jmfleming/KITTI'
NUSC_DIR = '/mnt/alpha/jmfleming/nuscenes_kitti'
CONFIG_KITTI = 'config/labels/semantic-kitti-all.yaml'
CONFIG_NUSC = 'config/labels/nuscenes_new.yaml'
ARCH_KITTI = 'config/arch/senet-2048p.yml'
NUM_CLASSES = 17

def build_parser(root, data, arch, sensor_override=None, sequences=None):
    sensor = sensor_override if sensor_override is not None else arch["dataset"]["sensor"]
    seqs = sequences if sequences is not None else data["split"]["valid"]
    return Parser(root=root, train_sequences=seqs, valid_sequences=seqs,
                  test_sequences=None, labels=data["labels"], color_map=data["color_map"],
                  learning_map=data["learning_map"], learning_map_inv=data["learning_map_inv"],
                  sensor=sensor, max_points=arch["dataset"]["max_points"],
                  batch_size=1, workers=4, gt=True, shuffle_train=False)

def nusc_sensor(arch):
    s = arch["dataset"]["sensor"].copy()
    s["fov_up"] = 10.0
    s["fov_down"] = -30.0
    s["img_prop"] = s["img_prop"].copy()
    s["img_prop"]["height"] = 32
    s["img_prop"]["width"] = 1024
    return s

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

def proto_miou(feats, lbls, base_protos, proto_lbls, proj, device, chunk=100000):
    preds = []
    for s in range(0, len(feats), chunk):
        hc = F.normalize(torch.sign(feats[s:s + chunk].to(device) @ proj), p=2, dim=1)
        sims = hc @ F.normalize(base_protos, p=2, dim=1).T
        preds.append(proto_lbls[sims.argmax(dim=1)].cpu())
    preds = torch.cat(preds, dim=0)
    return compute_miou(preds, lbls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_dir", type=str, default=KITTI_DIR)
    parser.add_argument("--nusc_dir", type=str, default=NUSC_DIR)
    parser.add_argument("--config_kitti", type=str, default=CONFIG_KITTI)
    parser.add_argument("--config_nusc", type=str, default=CONFIG_NUSC)
    parser.add_argument("--arch", type=str, default=ARCH_KITTI)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=100000)
    parser.add_argument("--path", type=str, required=True, help="KITTI-trained checkpoint dir")
    parser.add_argument("--method", type=str, default="supcon_vib_dglsspp_inputin_in_chan")
    parser.add_argument("--label", type=str, default="covshift")
    parser.add_argument("--out", type=str, default="robust_diagnostic/logs/nusc_cross_domain.json")
    args = parser.parse_args()

    DATA_K = yaml.safe_load(open(args.config_kitti, 'r'))
    DATA_N = yaml.safe_load(open(args.config_nusc, 'r'))
    ARCH = yaml.safe_load(open(args.arch, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    trainer = GenTrainer(ARCH, DATA_K, args.kitti_dir, args.path, path=args.path, method=args.method)
    model = trainer.model

    # KITTI clean: frozen prototypes + probe source (train-side anchors). Uses the
    # held-out scene 08 (same as the other diagnostics) so the KITTI gap is comparable.
    print("Extracting KITTI clean features...")
    fa, la = extract_features(model, build_parser(args.kitti_dir, DATA_K, ARCH,
                                                  sequences=['08']), device, args.frames)
    proj = get_hdc_projection(dim_in=fa.shape[1], dim_out=10000, device=device)
    base_protos, base_lbls = build_hdc_prototypes(fa, la, proj, device=device)

    clf = LogisticRegression(max_iter=1000)
    fit_n = min(100000, len(fa))
    clf.fit(fa[:fit_n].numpy(), la[:fit_n].numpy())

    # KITTI clean baseline: the extractor's own-domain gap (reference)
    print("KITTI clean zero-shot / oracle (reference gap)...")
    torch.manual_seed(42)
    perm = torch.randperm(len(fa))
    pool, pl = fa[perm[:args.pool_size]], la[perm[:args.pool_size]]
    val, vl = fa[perm[-args.val_size:]], la[perm[-args.val_size:]]
    kitti_zs = proto_miou(val, vl, base_protos, base_lbls, proj, device)
    orc_k, olbl_k = build_hdc_prototypes(pool, pl, proj, device=device)
    kitti_oracle = proto_miou(val, vl, orc_k, olbl_k, proj, device)
    lp_acc = float((torch.tensor(clf.predict(val.numpy())) == vl).float().mean().item())
    lp_miou = compute_miou(torch.tensor(clf.predict(val.numpy())), vl)

    # NuScenes: the cross-domain target
    print("Extracting NuScenes features (32-beam sensor)...")
    nusc = build_parser(args.nusc_dir, DATA_N, ARCH, sensor_override=nusc_sensor(ARCH))
    fn, ln = extract_features(model, nusc, device, args.frames)
    print(f"  NuScenes points {len(fn)}")
    torch.manual_seed(42)
    perm_n = torch.randperm(len(fn))
    pool_n, pln = fn[perm_n[:args.pool_size]], ln[perm_n[:args.pool_size]]
    val_n, vln = fn[perm_n[-args.val_size:]], ln[perm_n[-args.val_size:]]

    nusc_zs = proto_miou(val_n, vln, base_protos, base_lbls, proj, device)
    orc_n, olbl_n = build_hdc_prototypes(pool_n, pln, proj, device=device)
    nusc_oracle = proto_miou(val_n, vln, orc_n, olbl_n, proj, device)
    nlp_preds = torch.tensor(clf.predict(val_n.numpy()))
    nlp_acc = float((nlp_preds == vln).float().mean().item())
    nlp_miou = compute_miou(nlp_preds, vln)

    results = {
        'label': args.label,
        'method': args.method,
        'path': args.path,
        'kitti_clean': {'hdc_zs': kitti_zs, 'hdc_oracle': kitti_oracle,
                        'gap_oracle_zs': kitti_oracle - kitti_zs,
                        'lp_acc': lp_acc, 'lp_miou': lp_miou},
        'nuscenes': {'hdc_zs': nusc_zs, 'hdc_oracle': nusc_oracle,
                     'gap_oracle_zs': nusc_oracle - nusc_zs,
                     'lp_acc': nlp_acc, 'lp_miou': nlp_miou,
                     'n_points': int(len(fn))},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*90}")
    print(f"=== {args.label} on NuScenes (KITTI-trained weights) ===")
    print(f"{'':<14} {'hdc_zs':>8} {'hdc_orc':>8} {'gap':>7} {'lp_acc':>7} {'lp_miou':>8}")
    for tag, r in [('KITTI-clean', results['kitti_clean']), ('NuScenes', results['nuscenes'])]:
        print(f"{tag:<14} {r['hdc_zs']:>8.4f} {r['hdc_oracle']:>8.4f} {r['gap_oracle_zs']:>7.4f} "
              f"{r['lp_acc']:>7.4f} {r['lp_miou']:>8.4f}")
    print(f"\nSaved to {args.out}")
    print("\n=== WHAT TO LOOK FOR ===")
    print("TTA-relevant quantity: gap = oracle - zs on NuScenes vs on KITTI clean.")
    print("  - If the NuScenes gap is LARGE while KITTI's is small, the cov-shift")
    print("    extractor transferred poorly (frozen prototypes far from NuScenes) but")
    print("    the features retain recoverable structure -> TTA headroom that did not")
    print("    exist on KITTI.")
    print("  - If lp_miou (learned 128-d probe) on NuScenes is high, the continuous")
    print("    features DID transfer and the gap is in the HDC prototype binarization.")

if __name__ == "__main__":
    main()
