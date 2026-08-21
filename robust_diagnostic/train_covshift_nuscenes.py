"""train_covshift_nuscenes.py: train the COV-SHIFT feature extractor on NuScenes
(from scratch, in KITTI format), so it can later be evaluated on NuScenes-C.

The cov-shift recipe (supcon_vib_dglsspp_inputin_in_chan) is trained on the
NuScenes train split with the SAME loss stack and input normalization as the
KITTI extractor, but on NuScenes data:

  * DATA   = config/labels/nuscenes_new.yaml (its split.train has ~700 scenes;
             the 16-class taxonomy shared with SemanticKITTI via learning_map)
  * sensor = the 32-beam NuScenes projection (fov_up=10, fov_down=-30,
             H=32, W=1024) -- NuScenes scans have 32 beams, so the KITTI 64-beam
             projection would leave 75% of the image empty (the
             unsup_kitti-nuscenes.py setup)
  * datadir = the NuScenes dataset in KITTI format (default
             /mnt/alpha/jmfleming/nuscenes_kitti)
  * method = supcon_vib_dglsspp_inputin_in_chan (per-scan normalization on
             channels {0,4} + internal InstanceNorm + GMSIFC/LSCC, no SupCon)

Checkpoints are saved to <log_dir>/SENet after each epoch (the standard
GenTrainer format), so isotropy_diag / the full-dataset diag can load them by
pointing --path_b / --extractors at the resulting directory.

Usage:
  uv run python robust_diagnostic/train_covshift_nuscenes.py \
    --epochs 21 --cutoff 1.0 \
    --log_dir robust_diagnostic/logs/nusc_covshift_21ep
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch

from modules.gen_trainers import GenTrainer

CONFIG_ARCH = "config/arch/senet-2048p.yml"
CONFIG_LABELS_NUSCENES = "config/labels/nuscenes_new.yaml"
METHOD = "supcon_vib_dglsspp_inputin_in_chan"

def nuscenes_sensor(arch):
    """32-beam NuScenes projection (the unsup_kitti-nuscenes.py override)."""
    sensor = arch["dataset"]["sensor"].copy()
    sensor["fov_up"] = 10.0
    sensor["fov_down"] = -30.0
    sensor["img_prop"] = sensor["img_prop"].copy()
    sensor["img_prop"]["height"] = 32
    sensor["img_prop"]["width"] = 1024
    return sensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc_dir", type=str, default="/mnt/alpha/jmfleming/nuscenes_kitti",
                    help="NuScenes dataset in KITTI format (32-beam)")
    ap.add_argument("--nusc_labels", type=str, default=CONFIG_LABELS_NUSCENES)
    ap.add_argument("--arch", type=str, default=CONFIG_ARCH)
    ap.add_argument("--log_dir", type=str, default="robust_diagnostic/logs/nusc_covshift_21ep")
    ap.add_argument("--epochs", type=int, default=21)
    ap.add_argument("--cutoff", type=float, default=1.0,
                    help="fraction of the NuScenes train split per epoch (1.0 = full)")
    ap.add_argument("--resume", action="store_true",
                    help="load <log_dir>/SENet and continue training to --epochs total")
    ap.add_argument("--method", type=str, default=METHOD)
    args = ap.parse_args()

    DATA = yaml.safe_load(open(args.nusc_labels, "r"))
    ARCH = yaml.safe_load(open(args.arch, "r"))
    # Override the sensor for the 32-beam NuScenes projection BEFORE the parser
    # is built inside GenTrainer.
    ARCH["dataset"]["sensor"] = nuscenes_sensor(ARCH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(f"Training {args.method} on NuScenes ({args.nusc_dir}), "
          f"{args.epochs} epochs / {args.cutoff} cutoff")

    os.makedirs(args.log_dir, exist_ok=True)
    if args.resume:
        print(f"=== Resuming from {args.log_dir}/SENet ===")
        trainer = GenTrainer(ARCH, DATA, args.nusc_dir, args.log_dir,
                             path=args.log_dir, method=args.method,
                             cutoff_percent=args.cutoff)
    else:
        trainer = GenTrainer(ARCH, DATA, args.nusc_dir, args.log_dir,
                             method=args.method, cutoff_percent=args.cutoff)
    trainer.train(epochs=args.epochs)
    print(f"\nDone. Checkpoints in {args.log_dir}/SENet*")

if __name__ == "__main__":
    main()
