#!/bin/bash
# ==============================================================================
# Section 7.3 Ablation Studies Benchmark Suite
# ==============================================================================
# This script executes the online test-time adaptation ablation suite for
# Section 7.3, evaluating each core component of our proposed architecture
# in isolation to confirm their theoretical and empirical contributions.
#
# Ablations evaluated:
#   1. frozen:                  No adaptation baseline
#   2. full_method:             Our complete unified architecture
#   3. no_dual_gating:          Without geometric Mahalanobis gating (Dirichlet only)
#   4. no_temporal_consistency: Without Bayesian Momentum inertia (normalized weights)
#   5. no_inter_class_balance:  Without tau-prior frequency boundary shift
#   6. no_intra_class_balance:  Without IC4 active learning gradient scaling
#   7. no_gating:               Without uncertainty gating (uniform weighting)
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="ablation_comparisons_sweep.log"
PRETRAINED_PATH="logs/kitti_pretrain/hdc_sub.pth"
LOG_DIR="logs/ablation_comparisons"
CORRUPTIONS="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3

mkdir -p "$LOG_DIR"

{
    echo "=========================================================="
    echo "Starting Section 7.3 Method Ablation Benchmark Suite"
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
    uv run ablation_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --ablations default \
      --corruptions snow \
      --severity $SEV \
      --chunked \
      --reset_per_corruption \
      --dry_run \
      --log_dir logs/dry_run_ablation_compare

    echo ""
    echo "=========================================================="
    echo "Systematic Method Ablation Benchmark (Section 7.3)"
    echo "=========================================================="

    # --- 1. Run Complete Ablation Suite across all Corruptions ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/1] Running all ablation configurations across all corruptions..."
    echo "----------------------------------------------------------"
    uv run ablation_kitti-c.py \
      --pretrained_path "$PRETRAINED_PATH" \
      --log_dir "$LOG_DIR" \
      --corruptions "$CORRUPTIONS" \
      --severity "$SEV" \
      --chunked \
      --reset_per_corruption \
      --ablations all

    echo ""
    echo "=========================================================="
    echo "✅ Method Ablation Benchmark Suite Completed at: $(date)"
    echo "Summary results saved to: $LOG_DIR/global_ablation_results.json"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
