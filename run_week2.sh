#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

# Set this to "" to run the full sweeps, or "--dry_run" to verify nothing crashes first.
DRY_RUN_FLAG=""

{
    echo "=========================================="
    echo "Week 2: Multi-View TTA Consensus"
    echo "=========================================="
    echo "Testing Spatial Augmentations (Base, Yaw-Shifted, Depth-Scaled)"
    echo "Using IC4 (Epistemic Weighting) + Calibrated tau=-1.0 as the base configuration."
    echo ""

    TTA_METHODS=("bundle" "min_uncert" "mean_uncert")
    
    for tta in "${TTA_METHODS[@]}"; do
        echo "------------------------------------------"
        echo "Testing Multi-View Strategy: ${tta}"
        echo "------------------------------------------"
        uv run unsup_kitti-c.py \
            --method evidential_hdc_tta \
            --ic_method ic4 \
            --chunked \
            --reset_per_corruption \
            --tau -1.0 \
            --kappa 15.0 \
            --mv_tta ${tta} \
            ${DRY_RUN_FLAG} \
            --log_dir ../Logs/week2_tta_${tta}
    done

    echo "=========================================="
    echo "Week 2 Multi-View TTA tests complete."
    echo "=========================================="
} 2>&1 | tee week2_results.log
