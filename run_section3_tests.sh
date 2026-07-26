#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="section3_tests.log"

{
    echo "=========================================================="
    echo "Section 3 Dual-Uncertainty Gating Evaluation Suite"
    echo "=========================================================="
    echo "Started at: $(date)"
    echo "=========================================================="
    
    PANEL="beam_missing,wet_ground,motion_blur"
    
    echo ""
    echo "=========================================================="
    echo "Running Diagnostic Panel ($PANEL)"
    echo "=========================================================="
    
    # --- 1. Test 3.1: Decoupled Dual-Uncertainty Gating (AND vs OR) ---
    echo "----------------------------------------------------------"
    echo "[Test 1/6] Running gate_mode=epistemic (Dirichlet evidence baseline)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta veto_disagree --gate_mode epistemic --log_dir ../Logs/section3_gating/epistemic
    
    echo "----------------------------------------------------------"
    echo "[Test 2/6] Running gate_mode=geometric (128D Mahalanobis distance)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta veto_disagree --gate_mode geometric --log_dir ../Logs/section3_gating/geometric
    
    echo "----------------------------------------------------------"
    echo "[Test 3/6] Running gate_mode=and_gate (Logical AND intersection)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta veto_disagree --gate_mode and_gate --log_dir ../Logs/section3_gating/and_gate
    
    echo "----------------------------------------------------------"
    echo "[Test 4/6] Running gate_mode=or_gate (Logical OR union)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta veto_disagree --gate_mode or_gate --log_dir ../Logs/section3_gating/or_gate
    
    # --- 2. Test 3.2: Cross-View Orthogonality Hypothesis ---
    echo "----------------------------------------------------------"
    echo "[Test 5/6] Running gate_mode=or_gate with mv_tta=none (Single-View OR Gating)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta none --gate_mode or_gate --log_dir ../Logs/section3_gating/or_gate_single_view
    
    # --- 3. Test 3.3: Dynamic Geometric Thresholding (--dynamic_geom) ---
    echo "----------------------------------------------------------"
    echo "[Test 6/6] Running gate_mode=or_gate with dynamic_geom (Running batch variance)..."
    uv run unsup_kitti-c.py --corruptions $PANEL --method evidential_hdc_tta --tau -1.0 --mv_tta veto_disagree --gate_mode or_gate --dynamic_geom --log_dir ../Logs/section3_gating/or_gate_dyn

    echo ""
    echo "=========================================================="
    echo "Section 3 Evaluation Suite Completed Successfully at: $(date)"
    echo "=========================================================="
    
} 2>&1 | tee $LOG_FILE
