#!/bin/bash
set -e

echo "========================================="
echo "Starting Overnight Diagnostic Suite (V1-V7)"
echo "========================================="

echo "\n[1/3] RUN 1: Baseline (V1/V4 Instrumentation, Clean Physics)"
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground

echo "\n[2/3] RUN 2: Bug Reproduction (V6: The 1/t Annealing Test)"
# Disables post-step normalization to confirm if the 0.4840 Snow score is recovered online
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground \
  --reproduce_bug

echo "\n[3/3] RUN 3: Noise Floor Calibration (V7)"
# Runs with a different random seed to establish the bounds of metric variation
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground \
  --seed 999

echo "\n========================================="
echo "Overnight Diagnostics Complete!"
echo "========================================="
