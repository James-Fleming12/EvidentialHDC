#!/bin/bash
# run_m_methods.sh
# Comprehensive sweep script to validate and run the M-series components
# M-A: Per-class rotation cap
# M-B: Continuous gain control
# M-C: Uncertainty-loosened gate
# M-D: Always-on prior estimation

SEEDS="42,43,44"

echo "=== STAGE 1: The Additive Component Ladder ==="
echo "Running the full ladder across 3 seeds to isolate each component's contribution."
echo "Target: Compare this ladder against the +2.62 ceiling from G1."
uv run ablation_kitti-c.py --ablations methods --seeds $SEEDS | tee logs/m_series_methods.log

echo ""
echo "=== STAGE 3: Isolated Prior Test ==="
echo "Running M-D (Prior Only) without adaptation to verify the +11.7 inference-only gain on wet_ground."
uv run ablation_kitti-c.py --ablations prior --seeds $SEEDS | tee logs/m_d_prior_only.log

echo ""
echo "All runs initiated/completed. Please review logs/m_series_methods.log for the ladder results."
