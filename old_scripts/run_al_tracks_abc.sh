#!/usr/bin/env bash
# run_al_tracks_abc.sh: comprehensive Tracks A/B/C on COV-SHIFT (anchor).
#   A1 budget curve to high k, A2 adaptive allocation, B residual low-rank,
#   with fallback to high budget diagnosis if cheap tracks fail.
# Eval-only, ~1-2 min per condition on one GPU.
#
# Usage:
#   bash run_al_tracks_abc.sh 3                         # cov-shift ep10, all conds
#   bash run_al_tracks_abc.sh 3 ep10 "fog,crosstalk"    # subset

set -u
set -o pipefail
GPU="${1:-3}"
TAG="${2:-ep10}"
CONDS="${3:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, tag=$TAG, conds=$CONDS"

if [ "$TAG" = "ep10" ]; then
  METHOD="supcon_vib_dglsspp_inputin_in_chan"
  CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
elif [ "$TAG" = "ep21" ]; then
  METHOD="supcon_vib_dglsspp_inputin_in_chan"
  CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
else
  METHOD="supcon_vib_dglsspp_inputin_in_chan"
  CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
fi

echo "=== [Tracks A/B/C] $TAG [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_tracks_abc_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "tracks_$TAG" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_tracks_abc_$TAG.json" \
  2>&1 | tee "logs/al_tracks_abc_$TAG.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== TRACKS A/B/C OK ==="
  echo "Check logs/al_tracks_abc_$TAG.log: budget curve, adaptive, residual."
  echo "If flat-negative to 128, see the fallback: where the knee is and what per-k t_cos/w_cos bought."
else
  echo "=== TRACKS A/B/C FAILED (exit $RC) ==="
  exit $RC
fi
