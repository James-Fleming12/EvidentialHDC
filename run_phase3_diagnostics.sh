#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="phase3_diagnostics.log"

{
    echo "=========================================="
    echo "Phase 3 Diagnostics: MV-1 and MV-3 (tau=0 vs tau=-1)"
    echo "=========================================="
    echo "Testing bundle vs bundle_gentle vs baseline at tau=0.0 and tau=-1.0"
    
    # 0. MV-1: tau=0.0 sweeps
    echo "------------------------------------------"
    echo "Running baseline (tau=0.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta none --log_dir ../Logs/tau_0_base
    
    echo "------------------------------------------"
    echo "Running bundle (tau=0.0) [90 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle --log_dir ../Logs/tau_0_bundle
    
    echo "------------------------------------------"
    echo "Running bundle_gentle (tau=0.0) [11 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle_gentle --log_dir ../Logs/tau_0_bundle_gentle

    # 1. 3.2 Ablation: plain mean at tau=-1.0
    echo "------------------------------------------"
    echo "Running baseline (tau=-1.0, ic_method=none)"
    echo "This tests if IC4 is decoration and tau+dilution-fix is the hero."
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta none --log_dir ../Logs/tau_minus1_base

} 2>&1 | tee $LOG_FILE

chmod +x run_phase3_diagnostics.sh
