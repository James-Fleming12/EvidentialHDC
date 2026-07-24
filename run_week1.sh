#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# Set this to "" to run the full sweeps, or "--dry_run" to verify nothing crashes first.
DRY_RUN_FLAG=""

{
    echo "=========================================="
    echo "Week 1: Inter/Intra-Class Balancing on Calibrated Pseudo-labels"
    echo "=========================================="
    echo "We know tau=-1.0, kappa=15.0 solves the precision (Zero-Shot Calibration)."
    echo "Now we must unfreeze the adaptation using IC4 (Epistemic scaling) and XC2 (Geometric sub-clustering)."
    echo ""

    IC_METHODS=("ic4" "xc2")
    
    # We will test both unnormalized (Bayesian Momentum) and normalized (No Inertia)
    # to see which geometry interacts best with the structural IC/XC gradients.
    
    for ic in "${IC_METHODS[@]}"; do
        echo "------------------------------------------"
        echo "Testing ${ic} (Unnormalized / Bayesian Momentum)"
        echo "------------------------------------------"
        python unsup_kitti-c.py \
            --method evidential_hdc_tta \
            --ic_method ${ic} \
            --chunked \
            --reset_per_corruption \
            --tau -1.0 \
            --kappa 15.0 \
            ${DRY_RUN_FLAG} \
            --log_dir logs/week1_calibrated_${ic}_unnormalized

        echo "------------------------------------------"
        echo "Testing ${ic} (Normalized / No Inertia)"
        echo "------------------------------------------"
        python unsup_kitti-c.py \
            --method evidential_hdc_tta \
            --ic_method ${ic} \
            --chunked \
            --reset_per_corruption \
            --tau -1.0 \
            --kappa 15.0 \
            --normalize_weights \
            ${DRY_RUN_FLAG} \
            --log_dir logs/week1_calibrated_${ic}_normalized
    done

    echo "=========================================="
    echo "Week 1 Sweeps Complete."
    echo "=========================================="
} 2>&1 | tee week1_results.log
