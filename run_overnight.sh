#!/bin/bash
# run_overnight.sh
# Combines D1, D3, D4, and D5 diagnostics/tests.

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

SEEDS="42"
CORRUPTIONS="wet_ground,fog,crosstalk,incomplete_echo"
ALL_CORRUPTIONS="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"

echo "=== Running D1 & D4 Diagnostics + New Methods (Prior Switch & Adaptive Budget) ==="
# 'overnight' set includes: frozen, m_d_prior_only, m_d_prior_switch, m_a_adaptive_cap, m_abc_loosen
uv run python ablation_kitti-c.py --ablations overnight \
    --seeds $SEEDS \
    --corruptions $CORRUPTIONS \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/methods 2>&1 | tee logs/diagnostics_overnight_methods.log

echo "=== Running D3 Drift-Knee Sweep (wet_ground,crosstalk) ==="
uv run python ablation_kitti-c.py --ablations d3_sweep \
    --seeds $SEEDS \
    --corruptions wet_ground,crosstalk \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/d3 2>&1 | tee logs/diagnostics_overnight_d3.log

echo "=== Running D5 Stopping Test (All 8 corruptions, 3 Seeds) ==="
# Compares m_d_prior_switch against m_a_adaptive_cap across all corruptions
uv run python ablation_kitti-c.py --ablations m_d_prior_switch,m_a_adaptive_cap \
    --seeds 42,43,44 \
    --corruptions $ALL_CORRUPTIONS \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/d5 2>&1 | tee logs/diagnostics_overnight_d5.log

echo "=== Running T1 & T-LOOSE (Chunked) ==="
# legacy_val includes: legacy_frozen_t0, legacy_frozen_t1, legacy_loose_t0, legacy_loose_t1
uv run python ablation_kitti-c.py --ablations legacy_val \
    --seeds $SEEDS \
    --corruptions $ALL_CORRUPTIONS \
    --chunked --reset_per_corruption \
    --log_dir logs/diagnostics_overnight/legacy_chunked 2>&1 | tee logs/diagnostics_overnight_legacy_chunked.log

echo "=== Running T-DRIFT (Continual) ==="
# Run the loose adaptation without reset to show the structural collapse
uv run python ablation_kitti-c.py --ablations legacy_loose_t1 \
    --seeds $SEEDS \
    --corruptions wet_ground \
    --log_dir logs/diagnostics_overnight/legacy_continual 2>&1 | tee logs/diagnostics_overnight_legacy_continual.log

echo "Overnight run complete."