#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="phase3_intervention.log"

{
    echo "=========================================================="
    echo "Phase 3 Intervention: MV-2 Active Disagreement Veto & Conf-Pred"
    echo "=========================================================="
    echo "Testing veto_disagree vs conf_pred vs baseline at tau=0.0 and tau=-1.0"
    
    # 1. tau=0.0 Suite (Testing prior-free unsupervised disagreement intervention)
    echo "----------------------------------------------------------"
    echo "Running baseline (tau=0.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta none --log_dir ../Logs/phase3_interv/tau_0_base
    
    echo "----------------------------------------------------------"
    echo "Running veto_disagree (tau=0.0) [Active Disagreement Veto]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta veto_disagree --log_dir ../Logs/phase3_interv/tau_0_veto
    
    echo "----------------------------------------------------------"
    echo "Running conf_pred (tau=0.0) [Prediction-Path Consensus Hero]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta conf_pred --log_dir ../Logs/phase3_interv/tau_0_conf

    # 2. tau=-1.0 Suite (Testing synergy with prior calibration)
    echo "----------------------------------------------------------"
    echo "Running baseline (tau=-1.0)"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta none --log_dir ../Logs/phase3_interv/tau_minus1_base

    echo "----------------------------------------------------------"
    echo "Running veto_disagree (tau=-1.0) [Active Disagreement Veto]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta veto_disagree --log_dir ../Logs/phase3_interv/tau_minus1_veto

    echo "----------------------------------------------------------"
    echo "Running conf_pred (tau=-1.0) [Prediction-Path Consensus Hero]"
    uv run unsup_kitti-c.py --corruptions snow --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta conf_pred --log_dir ../Logs/phase3_interv/tau_minus1_conf

} 2>&1 | tee $LOG_FILE