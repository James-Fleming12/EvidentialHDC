#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "B1/B2: Chunked-Protocol Noise Floor (3 seeds)"
echo "=========================================="

SEEDS=(42 43 44)

for seed in "${SEEDS[@]}"; do
    echo "Running baseline with seed ${seed}..."
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method none \
        --chunked \
        --reset_per_corruption \
        --seed ${seed} \
        --log_dir logs/noise_floor_chunked_seed_${seed}
done

echo "=========================================="
echo "C1: Frozen tau sweep"
echo "=========================================="
# (tau sweep requires code changes which will be done later)

echo "Part C Complete."
