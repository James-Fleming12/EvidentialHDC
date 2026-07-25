#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="phase2_bundle_retry.log"

{
    echo "=========================================="
    echo "Phase 2 Retry: Feature Bundling Strategies"
    echo "=========================================="
    
    echo "------------------------------------------"
    echo "Step 0: bundle (tau=0.0) [90 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle --log_dir ../Logs/tau_0_bundle
    
    echo "------------------------------------------"
    echo "Step 0: bundle_moderate (tau=0.0) [22 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle_moderate --log_dir ../Logs/tau_0_bundle_moderate

    echo "------------------------------------------"
    echo "Step 0: bundle_gentle (tau=0.0) [11 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle_gentle --log_dir ../Logs/tau_0_bundle_gentle

} 2>&1 | tee $LOG_FILE
