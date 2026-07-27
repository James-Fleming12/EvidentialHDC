#!/bin/bash
# ==============================================================================
# run_baselines_compare.sh
# ==============================================================================
# Runs SemanticKITTI-C test-time adaptation evaluations for all methods in the 
# Section 7.2 Benchmark Table using the pretrained feature extractor and HDC model.
#
# Usage:
#   bash run_baselines_compare.sh [PRETRAINED_PATH] [LOG_DIR] [CORRUPTIONS] [SEVERITY]
#
# Example:
#   bash run_baselines_compare.sh logs/kitti_pretrain/hdc_sub.pth logs/baseline_comparisons fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor 3
# ==============================================================================

PRETRAINED_PATH=${1:-"logs/kitti_pretrain/hdc_sub.pth"}
LOG_DIR=${2:-"logs/baseline_comparisons"}
CORRUPTIONS=${3:-"fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"}
SEVERITY=${4:-3}

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "Starting Section 7.2 Method Comparisons Benchmark Suite"
echo "Pretrained Model: $PRETRAINED_PATH"
echo "Log Directory:    $LOG_DIR"
echo "Corruptions:      $CORRUPTIONS (Severity: $SEVERITY)"
echo "========================================================================"
echo ""

# 1. Run Frozen Baseline, D3CTTA, ConformalHDC, and HyperDUM
echo "[Pass 1/3] Running Frozen, D3CTTA, ConformalHDC, and HyperDUM baselines..."
python3 unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED_PATH" \
    --log_dir "$LOG_DIR" \
    --corruptions "$CORRUPTIONS" \
    --severity "$SEVERITY" \
    --method frozen,d3ctta,conformalhdc,hyperdum

# 2. Run This Method (Single-View / No MV Variant)
echo ""
echo "[Pass 2/3] Running This Method (Soft Dual-Weighting + BM-IC4, No MV)..."
python3 unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED_PATH" \
    --log_dir "$LOG_DIR" \
    --corruptions "$CORRUPTIONS" \
    --severity "$SEVERITY" \
    --method evidential_hdc_tta \
    --gate_mode soft_dual_weight \
    --ic_method ic4 \
    --tau -1.0 \
    --mv_tta none \
    --dynamic_geom

# 3. Run This Method + MV-TTA (veto_disagree Variant)
echo ""
echo "[Pass 3/3] Running This Method + MV-TTA (Soft Dual-Weighting + veto_disagree)..."
python3 unsup_kitti-c.py \
    --pretrained_path "$PRETRAINED_PATH" \
    --log_dir "$LOG_DIR" \
    --corruptions "$CORRUPTIONS" \
    --severity "$SEVERITY" \
    --method evidential_hdc_tta \
    --gate_mode soft_dual_weight \
    --ic_method ic4 \
    --tau -1.0 \
    --mv_tta veto_disagree \
    --dynamic_geom

echo ""
echo "========================================================================"
echo "✅ All benchmark runs completed!"
echo "Summary results saved to: $LOG_DIR/global_results.json"
echo "========================================================================"
