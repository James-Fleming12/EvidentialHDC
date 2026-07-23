#!/bin/bash
set -e

echo "========================================="
echo "Starting S2: Explicit Learning Rate Schedule"
echo "========================================="

echo -e "\n[1/1] RUN 1: Schedule = s2_equilibrium (Formalized Bug Reproduction)"
PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 uv run unsup_kitti-c.py \
  --method evidential_hdc_tta \
  --corruptions snow,wet_ground \
  --schedule s2_equilibrium

echo -e "\n========================================="
echo "Diagnostics Complete!"
echo "========================================="
