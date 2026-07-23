#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# Set this to "" to run the full sweeps, or "--dry_run" to verify nothing crashes first.
DRY_RUN_FLAG=""

# We wrap everything in a block and pipe it to tee so you have a single log file
{
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
            ${DRY_RUN_FLAG} \
            --log_dir logs/noise_floor_chunked_seed_${seed}
    done

    echo "=========================================="
    echo "C1: Frozen tau sweep (Logit Adjustment)"
    echo "=========================================="

    TAUS=(-1 -0.5 0 0.25 0.5 1.0)

    for tau in "${TAUS[@]}"; do
        echo "Running frozen sweep with tau=${tau}..."
        python unsup_kitti-c.py \
            --method frozen \
            --ic_method none \
            --chunked \
            --tau ${tau} \
            ${DRY_RUN_FLAG} \
            --log_dir logs/tau_sweep_tau_${tau}
    done

    echo "Part C Complete."
} 2>&1 | tee part_c_results.log
