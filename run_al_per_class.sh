#!/usr/bin/env bash
# run_al_per_class.sh: per-class mIoU breakdown on the FULL dataset (17-class
# setting) for the cov-shift ep10 extractor: pre-condition (clean) vs post
# (each KITTI-C condition) per-class IoU + recall + gap + confusion structure,
# so a new loss term can target the minority-class failure mode.
#
# Usage:
#   bash run_al_per_class.sh 3                # all 8 conditions
#   CONDS=fog,crosstalk bash run_al_per_class.sh 3
#   MAX_FRAMES=200 bash run_al_per_class.sh 3 # smoke test
#
# Output: robust_diagnostic/logs/al_per_class_ep10.json
#   clean: per-class support/IoU/recall/precision on clean KITTI
#   conds[<cond>]: pool_support, val_support, frozen_iou, ceiling_iou, gap,
#                  per-class recall, top-3 confusion targets per class

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
SUFFIX="ep10"
[ "$MAX_FRAMES" != "0" ] && SUFFIX="${SUFFIX}_f${MAX_FRAMES}"

echo "=== [per-class] $SUFFIX [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_per_class_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "perclass_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/al_per_class_${SUFFIX}.json" \
  2>&1 | tee "logs/al_per_class_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== PER-CLASS OK ==="
  echo "Check logs/al_per_class_${SUFFIX}.log:"
  echo "  - per-class IoU pre (clean) vs post (condition) vs ceiling"
  echo "  - per-class gap and top confusion targets (where minority points go)"
else
  echo "=== PER-CLASS FAILED (exit $RC) ==="
  exit $RC
fi
