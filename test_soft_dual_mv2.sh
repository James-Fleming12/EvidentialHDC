#!/bin/bash
# ==============================================================================
# Quick Verification: soft_dual_weight + Multi-View Addon (mv_tta=veto_disagree)
# ==============================================================================
# This script executes a fast evaluation of soft_dual_weight combined with
# our Multi-View Disagreement Veto (veto_disagree) across snow, beam_missing,
# and wet_ground at severity 3.
#
# Goal: Verify that the multi-view veto cleans up false-positive snowflake
# scatter on snow (restoring mIoU >= 0.5064) while preserving the massive
# +0.0468 mIoU breakthrough on wet_ground (mIoU ~ 0.5626).
# ==============================================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

LOG_FILE="soft_dual_mv2_test.log"
PANEL="snow,beam_missing,wet_ground"
METHOD="bm_ic4"
SEV=3
TAU="-1.0"
IC="ic4"

{
    echo "=========================================================="
    echo "Starting Quick Verification: soft_dual_weight + MV-2 Veto"
    echo "Method: $METHOD | Panel: $PANEL | Severity: $SEV | IC: $IC | Tau: $TAU"
    echo "Started at: $(date)"
    echo "=========================================================="

    # --- 1. Evaluate soft_dual_weight + mv_tta=veto_disagree ---
    echo ""
    echo "----------------------------------------------------------"
    echo "[Test 1/1] Evaluating soft_dual_weight with mv_tta=veto_disagree..."
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
      --log_dir logs/soft_dual_mv2_test

    echo ""
    echo "=========================================================="
    echo "Quick Verification Completed at: $(date)"
    echo "=========================================================="

} 2>&1 | tee $LOG_FILE
