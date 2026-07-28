#!/bin/bash
# ==============================================================================
# Section 7.2 / SOTA Competitor Deep-Dive: D3CTTA Comparison Benchmark Suite
# ==============================================================================
# This script executes an online test-time adaptation comparison between our
# proposed unified architecture (Soft Dual-Weighting + BM-IC4 + Temporal Consistency)
# and the updated D3CTTA baseline across all 8 SemanticKITTI-C corruptions.
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="d3ctta_comparison_sweep.log"
PRETRAINED_PATH="logs/kitti_pretrain/hdc_sub.pth"
LOG_DIR="logs/d3ctta_comparison"
CORRUPTIONS="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3
TAU="-1.0"
IC="ic4"
METHOD="evidential_hdc_tta"

mkdir -p "$LOG_DIR"

{
    echo "=========================================================="
    echo "Starting D3CTTA Deep-Dive Comparison Benchmark Suite"
    echo "Pretrained Model: $PRETRAINED_PATH"
    echo "Log Directory:    $LOG_DIR"
    echo "Corruptions:      $CORRUPTIONS (Severity: $SEV)"
    echo "Started at:       $(date)"
    echo "=========================================================="

    # --- 0. Verification Dry Run ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Step 0] Running Dry Run Verification (2 batches on snow-3)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --method d3ctta \
      --corruptions snow \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dry_run \
      --log_dir logs/dry_run_d3ctta_compare

    echo ""
    echo "=========================================================="
    echo "Comparative Method Adaptation Benchmark vs D3CTTA"
    echo "=========================================================="

    # --- 1. Run Frozen Baseline and D3CTTA ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/3] Running Frozen Baseline and D3CTTA..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --log_dir "$LOG_DIR" \
      --corruptions "$CORRUPTIONS" \
      --severity "$SEV" \
      --chunked \
      --reset_per_corruption \
      --method frozen,d3ctta

    # --- 2. Run This Method (Single-View / No MV Variant) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 2/3] Running This Method (Soft Dual-Weighting + BM-IC4, No MV)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --log_dir "$LOG_DIR" \
      --corruptions "$CORRUPTIONS" \
      --severity "$SEV" \
      --chunked \
      --reset_per_corruption \
      --method $METHOD \
      --gate_mode soft_dual_weight \
      --ic_method $IC \
      --tau $TAU \
      --mv_tta none \
      --dynamic_geom

    # --- 3. Run This Method + MV-TTA (veto_disagree Variant) ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 3/3] Running This Method + MV-TTA (Soft Dual-Weighting + veto_disagree)..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --log_dir "$LOG_DIR" \
      --corruptions "$CORRUPTIONS" \
      --severity "$SEV" \
      --chunked \
      --reset_per_corruption \
      --method $METHOD \
      --gate_mode soft_dual_weight \
      --ic_method $IC \
      --tau $TAU \
      --mv_tta veto_disagree \
      --dynamic_geom

    echo ""
    echo "=========================================================="
    echo "✅ D3CTTA Comparison Benchmark Suite Completed at: $(date)"
    echo "Summary results saved to: $LOG_DIR/global_results.json"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
