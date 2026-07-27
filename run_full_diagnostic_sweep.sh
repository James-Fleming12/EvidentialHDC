#!/bin/bash
# ==============================================================================
# Full Diagnostic Sweep & Offline Feature Dump Script (Calibrated Regime)
# ==============================================================================
# This script executes the complete 3x3 factorial sweep for Section 3,
# including the dry run verification, Oracle ceiling, and offline feature dumping.
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="full_diagnostic_sweep.log"
PANEL="snow,beam_missing,wet_ground"
METHOD="bm_ic4"
SEV=3
TAU="-1.0"
IC="ic4"

{
    echo "=========================================================="
    echo "Starting Full Diagnostic Sweep & Validation Suite (Calibrated tau=$TAU)"
    echo "Method: $METHOD | Panel: $PANEL | Severity: $SEV | IC: $IC"
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
      --corruptions snow \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dry_run \
      --dump_features \
      --mv_tta veto_disagree \
      --dynamic_geom \
      --log_dir logs/dry_run_test

    echo ""
    echo "=========================================================="
    echo "Part 1: Factorial Gating Sweep & Cross-View Disagreement"
    echo "=========================================================="

    # --- 1. Epistemic Gating (Baseline) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/6] Epistemic Gating + Single-View Baseline (mv_tta=none)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode epistemic \
      --dynamic_geom \
      --mv_tta none \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_epi_none

    # --- 2. Epistemic Gating + MV-2 Veto Disagreement ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 2/6] Epistemic Gating + MV-2 Veto Disagreement (mv_tta=veto_disagree)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode epistemic \
      --dynamic_geom \
      --mv_tta veto_disagree \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_epi_mv2

    # --- 3. Geometric Gating (Dynamic Geom) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 3/6] Geometric Gating (Dynamic Mahalanobis Variance)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode geometric \
      --dynamic_geom \
      --mv_tta none \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_geom_dyn

    # --- 4. OR-Gate (Epistemic OR Geometric) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 4/6] Logical OR-Gate (Epistemic OR Geometric)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode or_gate \
      --dynamic_geom \
      --mv_tta none \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_orgate

    # --- 5. AND-Gate (Epistemic AND Geometric) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 5/6] Logical AND-Gate (Epistemic AND Geometric)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode and_gate \
      --dynamic_geom \
      --mv_tta none \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_andgate

    # --- 6. Oracle Gating (Upper Ceiling) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 6/6] Oracle Gating (Upper Ceiling)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode oracle \
      --dynamic_geom \
      --mv_tta none \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --log_dir logs/sweep_oracle

    echo ""
    echo "=========================================================="
    echo "Part 2: Offline Feature Dump for 12-Signal Breadth Probe"
    echo "=========================================================="
    echo "----------------------------------------------------------"
    echo "[Feature Dump] Exporting per-point 12-signal tensors..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --method $METHOD \
      --ic_method $IC \
      --tau $TAU \
      --gate_mode epistemic \
      --dynamic_geom \
      --mv_tta veto_disagree \
      --corruptions $PANEL \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dump_features \
      --log_dir logs/d5_d6_features_dump

    echo ""
    echo "=========================================================="
    echo "Full Diagnostic Sweep Completed Successfully at: $(date)"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
