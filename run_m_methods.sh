#!/bin/bash
# run_m_methods.sh
# Comprehensive sweep script to validate and run the M-series components
# M-A: Per-class rotation cap
# M-B: Continuous gain control
# M-C: Uncertainty-loosened gate
# M-D: Always-on prior estimation

SEEDS="42 43 44"

echo "=== STAGE 1: Dry Run Validation ==="
# Test the new configurations with --dry_run to ensure no syntax/runtime crashes on the pipeline
uv run ablation_kitti-c.py --ablations m_a_cap m_ab_gain m_abc_loosen m_abcd_prior m_d_prior_only --dry_run
if [ $? -ne 0 ]; then
    echo "Dry run failed! Please check syntax and logic errors."
    exit 1
fi
echo "Dry run successful!"

echo ""
echo "=== STAGE 2: The Additive Component Ladder ==="
echo "Running the full ladder across 3 seeds to isolate each component's contribution."
echo "Target: Compare this ladder against the +2.62 ceiling from G1."
uv run ablation_kitti-c.py --sets methods --seeds $SEEDS | tee logs/m_series_methods.log

echo ""
echo "=== STAGE 3: Isolated Prior Test ==="
echo "Running M-D (Prior Only) without adaptation to verify the +11.7 inference-only gain on wet_ground."
uv run ablation_kitti-c.py --sets prior --seeds $SEEDS | tee logs/m_d_prior_only.log

echo ""
echo "All runs initiated/completed. Please review logs/m_series_methods.log for the ladder results."
