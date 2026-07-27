#!/bin/bash
# ==============================================================================
# Dual Gating Architecture Comparative Adaptation Sweep
# ==============================================================================
# This script executes the online comparative adaptation sweep for Section 4.2
# and Section 4.3, evaluating the proposed asymmetric dual-gating architectures.
#
# All general candidate architectures (rescue_gate, ellipsoid_gate,
# soft_dual_weight) are tested on the TYPICAL NON-MULTI-VIEW VARIANT
# (mv_tta=none) to prove core gating capability.
# Candidate D (view_var_gate) is specifically designed as a multi-view gating
# mechanism and is therefore evaluated under multi-view (mv_tta=veto_disagree).
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="dual_gating_sweep.log"
PANEL="snow,beam_missing,wet_ground"
METHOD="bm_ic4"
SEV=3
TAU="-1.0"
IC="ic4"

{
    echo "=========================================================="
    echo "Starting Dual Gating Architecture Sweep (Core Non-Multi-View Variant)"
    echo "Method: $METHOD | Panel: $PANEL | Severity: $SEV | IC: $IC | Tau: $TAU"
    echo "Started at: $(date)"
    echo "=========================================================="

    # --- 0. Verification Dry Run ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Step 0] Running Dry Run Verification (2 batches on snow-3)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode soft_dual_weight \
      --mv_tta none \
      --dynamic_geom \
      --corruptions snow \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dry_run \
      --log_dir logs/dry_run_dual_gate

    echo ""
    echo "=========================================================="
    echo "Comparative Gating Architecture Sweep (Section 4.2)"
    echo "=========================================================="

    # --- 1. Epistemic Gating (Baseline Reference - Non-Multi-View Variant) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/6] Non-Multi-View Baseline: Epistemic Gating (mv_tta=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode epistemic \
      --mv_tta none \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_1_epi_nomv

    # --- 2. Candidate A: Conditional High-Precision Geometric Rescue ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 2/6] Candidate A: Conditional Rescue (rescue_gate, mv_tta=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode rescue_gate \
      --mv_tta none \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_2_rescue_nomv

    # --- 3. Candidate B: Adaptive 2D Ellipsoidal Decision Boundary ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 3/6] Candidate B: Ellipsoidal Boundary (ellipsoid_gate, mv_tta=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode ellipsoid_gate \
      --mv_tta none \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_3_ellipsoid_nomv

    # --- 4. Candidate C: Dynamic Multi-Metric Momentum Modulation ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 4/6] Candidate C: Multi-Metric Modulation (soft_dual_weight, mv_tta=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode soft_dual_weight \
      --mv_tta none \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_4_soft_dual_nomv

    # --- 5. Candidate D: Cross-View Softmax Variance Gating (Multi-View Specific) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 5/6] Candidate D (Multi-View Specific): Cross-View Variance (view_var_gate, mv_tta=veto_disagree)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode view_var_gate \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_5_view_var_mv2

    # --- 6. Calibration Ablation: soft_dual_weight in Uncalibrated Regime ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 6/6] Calibration Ablation: Uncalibrated Regime (soft_dual_weight, tau=None, ic_method=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method bm \
      --ic_method none \
      --gate_mode soft_dual_weight \
      --mv_tta none \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_6_uncalibrated_soft_dual

    echo ""
    echo "=========================================================="
    echo "Dual Gating Sweep Completed Successfully at: $(date)"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
