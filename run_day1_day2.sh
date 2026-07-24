#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# Set this to "" to run the full sweeps, or "--dry_run" to verify nothing crashes first.
DRY_RUN_FLAG=""

{
    echo "=========================================="
    echo "Day 1 (C-b & C-c): Re-baseline on frozen tau=-1 & TTA with tau=-1"
    echo "=========================================="
    
    # Baseline with no prior (logs initial TP/FP/FN for C-b)
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method none \
        --chunked \
        --reset_per_corruption \
        ${DRY_RUN_FLAG} \
        --log_dir logs/day1_baseline_tau_0
        
    # Baseline with tau=-1 in the pseudo-label path (True TTA with Prior)
    # This also logs the C-b TP/FP/FN decomposition for tau=-1!
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method none \
        --chunked \
        --reset_per_corruption \
        --tau -1.0 \
        ${DRY_RUN_FLAG} \
        --log_dir logs/day1_baseline_tau_minus1
        
    echo "=========================================="
    echo "Day 2 (C-a): 2D kappa/tau sweep (Frozen)"
    echo "=========================================="
    
    TAUS=(-0.5 -1.0 -1.5 -2.0 -3.0)
    KAPPAS=(5.0 15.0 50.0)
    
    for tau in "${TAUS[@]}"; do
        for kappa in "${KAPPAS[@]}"; do
            echo "Running frozen sweep with tau=${tau}, kappa=${kappa}..."
            python unsup_kitti-c.py \
                --method frozen \
                --ic_method none \
                --chunked \
                --reset_per_corruption \
                --tau ${tau} \
                --kappa ${kappa} \
                ${DRY_RUN_FLAG} \
                --log_dir logs/day2_tau_sweep_tau_${tau}_kappa_${kappa}
        done
    done

    echo "=========================================="
    echo "Day 2 (C-d): Kill the Double Prior (Unnormalized vs Normalized TTA)"
    echo "=========================================="

    # We already have Unnormalized tau=0 and Unnormalized tau=-1 from Day 1!
    # So we only need to run Normalized tau=0 and Normalized tau=-1.

    echo "Running Normalized TTA with tau=0 (Pure Gradient, No Prior)..."
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method none \
        --chunked \
        --reset_per_corruption \
        --tau 0.0 \
        --normalize_weights \
        ${DRY_RUN_FLAG} \
        --log_dir logs/day2_normalized_tau_0

    echo "Running Normalized TTA with tau=-1 (Pure Gradient + Explicit Prior)..."
    python unsup_kitti-c.py \
        --method evidential_hdc_tta \
        --ic_method none \
        --chunked \
        --reset_per_corruption \
        --tau -1.0 \
        --normalize_weights \
        ${DRY_RUN_FLAG} \
        --log_dir logs/day2_normalized_tau_minus1

    echo "Day 1 and Day 2 Sweeps Complete."
} 2>&1 | tee day1_day2_results.log
