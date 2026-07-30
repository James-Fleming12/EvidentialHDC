#!/bin/bash
# run_redo.sh

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

SEEDS="42"
CORRUPTIONS="wet_ground,fog,crosstalk,incomplete_echo"
ALL_CORRUPTIONS="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"

echo "=== Rerunning Stage 1 (Frozen Prior Methods) ==="
uv run python ablation_kitti-c.py --ablations m_d_prior_only,m_d_prior_switch,m_d_prior_ramp,m_d_prior_inverse \
    --seeds $SEEDS \
    --corruptions $CORRUPTIONS \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/methods_fix 2>&1 | tee logs/diagnostics_overnight_methods_fix.log

echo "=== Rerunning Stage 3 (Prior Switch D5 Test) ==="
uv run python ablation_kitti-c.py --ablations m_d_prior_switch \
    --seeds $SEEDS \
    --corruptions $ALL_CORRUPTIONS \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/d5_fix 2>&1 | tee logs/diagnostics_overnight_d5_fix.log

echo "=== Rerunning Stage 5 (T-DRIFT) ==="
uv run python ablation_kitti-c.py --ablations legacy_loose_t1 \
    --seeds $SEEDS \
    --corruptions wet_ground \
    --log_dir logs/diagnostics_overnight/legacy_continual 2>&1 | tee logs/diagnostics_overnight_legacy_continual.log

echo "Catch-up complete."