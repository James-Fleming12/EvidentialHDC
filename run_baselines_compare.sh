#!/bin/bash
# ==============================================================================
# Section 7.2 Method Comparisons Benchmark Suite
# ==============================================================================
# This script executes the online test-time adaptation benchmark suite for
# Section 7.2, evaluating all baselines (Frozen, D3CTTA, ConformalHDC, HyperDUM)
# and our proposed SOTA architecture (Soft Dual-Weighting + BM-IC4, with and without
# multi-view disagreement veto).
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="baseline_comparisons_sweep.log"
PRETRAINED_PATH="logs/kitti_pretrain/hdc_sub.pth"
LOG_DIR="logs/baseline_comparisons"
CORRUPTIONS="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3
TAU="-1.0"
IC="ic4"
METHOD="evidential_hdc_tta"

mkdir -p "$LOG_DIR"

{
    echo "=========================================================="
    echo "Starting Section 7.2 Method Comparisons Benchmark Suite"
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
      --log_dir logs/dry_run_baseline_compare

    echo ""
    echo "=========================================================="
    echo "Comparative Method Adaptation Benchmark (Section 7.2)"
    echo "=========================================================="

    # --- 1. Run Frozen Baseline, D3CTTA, ConformalHDC, and HyperDUM ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/3] Running Frozen, D3CTTA, ConformalHDC, and HyperDUM baselines..."
    echo "----------------------------------------------------------"
    uv run unsup_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --log_dir "$LOG_DIR" \
      --corruptions "$CORRUPTIONS" \
      --severity "$SEV" \
      --chunked \
      --reset_per_corruption \
      --method frozen,d3ctta,conformalhdc,hyperdum

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
    echo "✅ Method Comparisons Benchmark Suite Completed at: $(date)"
    echo "Summary results saved to: $LOG_DIR/global_results.json"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
