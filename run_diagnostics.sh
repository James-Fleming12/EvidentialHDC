#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

echo "=== Killing stale contaminated records ==="
rm -f logs/ablation_v2/records.json

echo "=== D-STANDARD & D-POISON (T3A Textbook Prototype Update & ConformalHDC Variants) ==="
uv run python ablation_kitti-c.py \
    --ablations standard_t3a,conformalhdc,conformalhdc_10k \
    --seeds 42,43,44 \
    --corruptions fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor \
    --severity 3 \
    --chunked --reset_per_corruption \
    --log_dir logs/d_standard \
    2>&1 | tee logs/diagnostics_d_standard.log

echo "=== D-CEILING-CLEAN (Prior Only on 8 Corruptions) ==="
uv run python ablation_kitti-c.py \
    --ablations m_d_prior_only \
    --seeds 42 \
    --corruptions fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor \
    --severity 3 \
    --chunked --reset_per_corruption \
    --log_dir logs/d_ceiling \
    2>&1 | tee logs/diagnostics_d_ceiling.log
