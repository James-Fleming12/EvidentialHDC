#!/bin/bash
# run_m_methods.sh
# Comprehensive sweep script to validate and run the M-series components
# M-A: Per-class rotation cap
# M-B: Continuous gain control
# M-C: Uncertainty-loosened gate
# M-D: Always-on prior estimation

SEEDS="42"

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Move the contaminated file aside
mv logs/ablation_v2/records.json logs/ablation_v2/records.json.contaminated 2>/dev/null || true

echo "=== STAGE 1: The Additive Component Ladder ==="
echo "Running the full ladder across 1 seed to isolate each component's contribution."
echo "Target: Compare this ladder against the +2.62 ceiling from G1."
uv run python ablation_kitti-c.py --ablations methods --seeds $SEEDS --chunked --reset_per_corruption --log_dir logs/m_series_single/methods 2>&1 | tee logs/m_series_single_methods.log

echo ""
echo "=== STAGE 2: Isolated Prior Test ==="
echo "Running M-D (Prior Only) without adaptation to verify the +11.7 inference-only gain on wet_ground."
uv run python ablation_kitti-c.py --ablations prior --seeds $SEEDS --chunked --reset_per_corruption --log_dir logs/m_series_single/prior 2>&1 | tee logs/m_series_single_prior.log

echo ""
echo "All runs initiated/completed. Please review logs/m_series_single_methods.log for the ladder results."
