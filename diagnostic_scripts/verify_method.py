"""verify_method.py: for a checkpoint dir + candidate method string, build the
model via GenTrainer and report (a) what architecture the method actually builds
and (b) whether the checkpoint's state_dict matches it (missing/unexpected keys).

The decisive check for the HyperLiDAR question: whichever (checkpoint, method)
pair loads with 0 missing keys is the real one.
  * logs/kitti_pretrain      + method=baseline       (plain Trainer CENET)
  * logs/med_pretrain_supcon_vib + method=supcon_vib (SupCon+VIB)
Loads cleanly = that checkpoint was trained with that method.

Usage:
  uv run python diagnostic_scripts/verify_method.py logs/kitti_pretrain baseline
  uv run python diagnostic_scripts/verify_method.py logs/med_pretrain_supcon_vib supcon_vib
"""
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch

from modules.gen_trainers import GenTrainer

ap = argparse.ArgumentParser()
ap.add_argument("ckpt_dir", type=str)
ap.add_argument("method", type=str)
ap.add_argument("--config", type=str, default="config/labels/semantic-kitti-all.yaml")
ap.add_argument("--arch", type=str, default="config/arch/senet-2048p.yml")
args = ap.parse_args()

DATA = yaml.safe_load(open(args.config))
ARCH = yaml.safe_load(open(args.arch))

arch = copy.deepcopy(ARCH)
trainer = GenTrainer(arch, DATA, "/mnt/alpha/jmfleming/KITTI", args.ckpt_dir,
                     path=args.ckpt_dir, method=args.method)
model = trainer.model
n = sum(p.numel() for p in model.parameters())
print(f"method={args.method} ckpt={args.ckpt_dir}")
print(f"  built model params: {n / 1e6:.3f} M")
print(f"  model.input_in={getattr(model, 'input_in', None)}")
print(f"  has logvar_head: {getattr(model, 'logvar_head', None) is not None}")
print(f"  has conv_corr:   {getattr(model, 'conv_corr', None) is not None}")
print(f"  corr_dim={getattr(model, 'corr_dim', None)}")

sd = torch.load(os.path.join(args.ckpt_dir, "SENet"), map_location="cpu")["state_dict"]
model_sd = model.state_dict()
missing = [k for k in model_sd if k not in sd]
unexpected = [k for k in sd if k not in model_sd]
print(f"  missing keys (in model, not in ckpt): {len(missing)}  first: {missing[:5]}")
print(f"  unexpected keys (in ckpt, not in model): {len(unexpected)}  first: {unexpected[:5]}")
if not missing and not unexpected:
    print("  >>> CLEAN LOAD: this checkpoint was trained with this method")
else:
    print("  >>> NOT a clean load: checkpoint/method mismatch")
