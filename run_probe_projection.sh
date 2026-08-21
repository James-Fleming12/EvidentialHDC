#!/usr/bin/env bash
# run_probe_projection.sh: HDC projection variants for linear separability and
# mIoU, on the frozen cov-shift ep10 extractor (eval-only).
#
# Tests whether the fixed random +-1 HDC projection R is leaving patterns on the
# table: gaussian/sparse/ternary projections, whitened / within-class-whitened /
# rotated feature transforms, and dim variants (5k/20k/concat), measured by the
# R4 linear-probe mIoU + minority per-class IoU + code statistics (dead-frac,
# hamming, class separability). Also reports a label-free dynamic-selection
# proxy (highest code Hamming) vs the oracle-best projection per condition.
#
# Usage:
#   bash run_probe_projection.sh 3                 # all 8 conditions, 11 variants
#   CONDS=fog,wet_ground bash run_probe_projection.sh 3
#   MAX_FRAMES=200 bash run_probe_projection.sh 3  # smoke test
#
# Output: robust_diagnostic/logs/probe_projection_ep10.json
#   conds[<cond>][<variant>] = {frozen, ceiling, gap, minority_iou{...},
#                               dead_frac, hamming, sep, dim}

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

echo "=== [projection diag] $SUFFIX [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_projection_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "proj_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/probe_projection_${SUFFIX}.json" \
  2>&1 | tee "logs/probe_projection_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== PROJECTION DIAG OK ==="
  echo "Check logs/probe_projection_${SUFFIX}.log:"
  echo "  - frozen/ceiling mIoU per projection variant per condition"
  echo "  - minority per-class IoU + dead-frac/hamming/separability stats"
  echo "  - oracle-best vs hamming-proxy dynamic selection"
else
  echo "=== PROJECTION DIAG FAILED (exit $RC) ==="
  exit $RC
fi
