"""verify_ckpt.py: fingerprint a checkpoint's architecture to determine what
method it was actually trained with.

Resolves which "HyperLiDAR" checkpoint is authoritative. The HyperLiDAR paper
uses CENET (a plain supervised projection-based extractor, p=128), which the
codebase trains via the plain `Trainer()` class (unsup_main.py:115) and saves
as `SENet_valid_best`. Two candidate checkpoints exist:
  * logs/kitti_pretrain      (the plain-Trainer CENET extractor)
  * logs/med_pretrain_supcon_vib (the supcon_vib SupCon+VIB extractor, the
                                  previously-claimed "HyperLiDAR baseline")
This script prints architecture fingerprints so the two can be told apart:
param count, presence of logvar_head / conv_corr / adaptor / aux heads, and
the semantic_output classes.

Usage:
  uv run python diagnostic_scripts/verify_ckpt.py logs/kitti_pretrain
  uv run python diagnostic_scripts/verify_ckpt.py logs/med_pretrain_supcon_vib
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt_dir", type=str, help="checkpoint dir (contains SENet or SENet_valid_best)")
args = ap.parse_args()

for name in ("SENet", "SENet_valid_best"):
    p = os.path.join(args.ckpt_dir, name)
    if not os.path.exists(p):
        print(f"  {name} NOT FOUND in {args.ckpt_dir}")
        continue
    w = torch.load(p, map_location="cpu")
    sd = w["state_dict"]
    print(f"checkpoint: {p}")
    print(f"  top-level keys: {sorted(w.keys())}")
    print(f"  epoch: {w.get('epoch')}")

    def has(prefix):
        return any(k.startswith(prefix) for k in sd)

    print(f"  semantic_output.bias shape: {tuple(sd.get('semantic_output.bias', torch.tensor([-1])).shape)}")
    print(f"  has logvar_head   (SupCon+VIB): {has('logvar_head')}")
    print(f"  has conv_corr     (corr branch): {has('conv_corr')}")
    print(f"  has adaptor       (TTA adaptor): {has('adaptor')}")
    print(f"  has aux_head1     (aux losses) : {has('aux_head1')}")
    print(f"  has conv_1        (bottleneck) : {has('conv_1')}")
    n = sum(v.numel() for v in sd.values())
    print(f"  total params: {n / 1e6:.3f} M")
    break
