#!/bin/bash
# ==============================================================================
# Dual Gating Architecture Comparative Adaptation Sweep
# ==============================================================================
# This script executes the online comparative adaptation sweep for Section 4.2
# and Section 4.3 (Synergistic Ablation Study), evaluating the proposed
# asymmetric dual-gating architectures against the epistemic baseline.
#
# All tests are conducted in the calibrated regime (tau=-1.0, ic4) with
# Multi-View Disagreement Veto (mv_tta=veto_disagree) active to preserve
# our validated Phase 2 and Section 3 synergistic mechanisms.
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
    echo "Starting Dual Gating Architecture Sweep & Ablation Suite"
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
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions snow \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dry_run \
      --log_dir logs/dry_run_dual_gate

    echo ""
    echo "=========================================================="
    echo "Part 1: Comparative Gating Architecture Sweep (Section 4.2)"
    echo "=========================================================="

    # --- 1. Epistemic Gating + MV-2 Veto (Baseline Reference) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/5] Baseline Reference: Epistemic Gating + MV-2 Veto..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode epistemic \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_sweep_1_epi_mv2

    # --- 2. Candidate A: Conditional High-Precision Geometric Rescue (Cascade) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 2/5] Candidate A: Conditional Geometric Rescue (rescue_gate)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode rescue_gate \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_sweep_2_rescue_gate

    # --- 3. Candidate B: Adaptive 2D Ellipsoidal Decision Boundary (Quadratic) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 3/5] Candidate B: Ellipsoidal Decision Boundary (ellipsoid_gate)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode ellipsoid_gate \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_sweep_3_ellipsoid_gate

    # --- 4. Candidate C: Dynamic Multi-Metric Momentum Modulation (Linear Ramp) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 4/5] Candidate C: Dynamic Multi-Metric Modulation (soft_dual_weight)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode soft_dual_weight \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/dual_sweep_4_soft_dual_weight

    # --- 5. Candidate D: Cross-View Softmax Variance Gating (V2 Goldmine) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 5/5] Candidate D: Cross-View Softmax Variance Gating (view_var_gate)..."
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
      --log_dir logs/dual_sweep_5_view_var_gate

    echo ""
    echo "=========================================================="
    echo "Part 2: Synergistic Ablation Study on soft_dual_weight (Section 4.3)"
    echo "=========================================================="

    # --- Ablation 1: Uncalibrated Regime (no tau, no IC) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Ablation 1/2] soft_dual_weight in Uncalibrated Regime (tau=None, ic_method=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method bm \
      --ic_method none \
      --gate_mode soft_dual_weight \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/ablation_soft_dual_uncalibrated

    # --- Ablation 2: Single-View Regime (mv_tta=none) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Ablation 2/2] soft_dual_weight without MV-2 Veto (mv_tta=none)..."
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
      --log_dir logs/ablation_soft_dual_nomv

    echo ""
    echo "=========================================================="
    echo "Dual Gating Sweep & Ablation Suite Completed at: $(date)"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
