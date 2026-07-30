#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

echo "=== T0a: Re-running Prior Panel with Guard ==="
uv run python ablation_kitti-c.py \
    --ablations m_d_prior_only,m_d_prior_switch,m_d_prior_ramp,m_d_prior_inverse \
    --seeds 42 \
    --corruptions wet_ground,fog,crosstalk,incomplete_echo \
    --severity 3 \
    --chunked --reset_per_corruption \
    2>&1 | tee diagnostics_repro_t0a.log

echo "=== T0b: Deliberate Prior Boost ==="
uv run python ablation_kitti-c.py \
    --ablations m_d_prior_boosted \
    --seeds 42 \
    --corruptions wet_ground \
    --severity 3 \
    --chunked --reset_per_corruption \
    2>&1 | tee diagnostics_repro_t0b.log
