#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="phase2_diagnostics.log"

{
    echo "=========================================="
    echo "Phase 2 Diagnostics (Multi-View Tests & Ablations)"
    echo "=========================================="
    
    echo "------------------------------------------"
    echo "Step 0: baseline (tau=0.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta none --log_dir ../Logs/tau_0_base
    
    echo "------------------------------------------"
    echo "Step 0: bundle (tau=0.0) [90 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle --log_dir ../Logs/tau_0_bundle
    
    echo "------------------------------------------"
    echo "------------------------------------------"
    echo "Step 0: bundle_moderate (tau=0.0) [22 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle_moderate --log_dir ../Logs/tau_0_bundle_moderate

    echo "Step 0: bundle_gentle (tau=0.0) [11 degree yaw]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta bundle_gentle --log_dir ../Logs/tau_0_bundle_gentle

    echo "------------------------------------------"
    echo "Step 1: vote_pred (tau=0.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta vote_pred --log_dir ../Logs/tau_0_vote_pred

    echo "------------------------------------------"
    echo "Step 1: conf_pred (tau=0.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta conf_pred --log_dir ../Logs/tau_0_conf_pred

    echo "------------------------------------------"
    echo "Step 1: vote_pred (tau=-1.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta vote_pred --log_dir ../Logs/tau_minus1_vote_pred

    echo "------------------------------------------"
    echo "Step 1: conf_pred (tau=-1.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta conf_pred --log_dir ../Logs/tau_minus1_conf_pred

    echo "------------------------------------------"
    echo "Step 2: 3.1 Ablation - IC4 at tau=0.0 (Testing Step Dilution guard)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method ic4 --tau 0.0 --mv_tta none --log_dir ../Logs/tau_0_ic4

    echo "------------------------------------------"
    echo "Step 2: 3.2 Ablation - Plain Mean at tau=-1.0 (Testing if IC4/XC2 are decoration)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta none --log_dir ../Logs/tau_minus1_base

} 2>&1 | tee $LOG_FILE

chmod +x run_phase2_diagnostics.sh
