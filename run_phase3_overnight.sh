#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="phase3_overnight.log"

{
    echo "=========================================================="
    echo "Phase 3 Overnight Intervention Suite: MV-2 & Prior Calibration"
    echo "=========================================================="
    echo "Started at: $(date)"
    echo "=========================================================="
    
    # -------------------------------------------------------------------------
    # Full Diagnostic & Challenging Panel Evaluation
    # Corruptions: beam_missing, wet_ground, fog, motion_blur
    # -------------------------------------------------------------------------
    PANEL="beam_missing,wet_ground,fog,motion_blur"
    
    echo ""
    echo "=========================================================="
    echo "Running Diagnostic Panel ($PANEL)"
    echo "=========================================================="
    
    # --- 1. tau=0.0 Suite (Prior-free unsupervised disagreement intervention) ---
    echo "----------------------------------------------------------"
    echo "[Panel Run 1/6] Running baseline (tau=0.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta none --log_dir ../Logs/phase3_interv/tau_0_base
    
    echo "----------------------------------------------------------"
    echo "[Panel Run 2/6] Running veto_disagree (tau=0.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta veto_disagree --log_dir ../Logs/phase3_interv/tau_0_veto
    
    echo "----------------------------------------------------------"
    echo "[Panel Run 3/6] Running conf_pred (tau=0.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau 0.0 --mv_tta conf_pred --log_dir ../Logs/phase3_interv/tau_0_conf
    
    # --- 2. tau=-1.0 Suite (Synergy with majority amplifier prior calibration) ---
    echo "----------------------------------------------------------"
    echo "[Panel Run 4/6] Running baseline (tau=-1.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta none --log_dir ../Logs/phase3_interv/tau_minus1_base
    
    echo "----------------------------------------------------------"
    echo "[Panel Run 5/6] Running veto_disagree (tau=-1.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta veto_disagree --log_dir ../Logs/phase3_interv/tau_minus1_veto
    
    echo "----------------------------------------------------------"
    echo "[Panel Run 6/6] Running conf_pred (tau=-1.0) on $PANEL..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --ic_method none --tau -1.0 --mv_tta conf_pred --log_dir ../Logs/phase3_interv/tau_minus1_conf

    echo ""
    echo "=========================================================="
    echo "Phase 3 Overnight Suite Completed Successfully at: $(date)"
    echo "=========================================================="
    
} 2>&1 | tee $LOG_FILE
