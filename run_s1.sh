#!/bin/bash
set -e

echo "========================================="
echo "Starting S1: Explicit Learning Rate Schedules"
echo "========================================="

echo "\n[1/2] RUN 1: Schedule = 1/t (Decaying Learning Rate)"
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground \
  --schedule 1/t

echo "\n[2/2] RUN 2: Schedule = cosine (Cosine Annealing)"
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground \
  --schedule cosine

echo "\n========================================="
echo "S1 Diagnostics Complete!"
echo "========================================="
