#!/usr/bin/env bash
# run_al_b_feature.sh: B decoder + feature-space potential on COV-SHIFT (frozen extractor kept).
# All eval-only, ~1 min per condition. Tests whether the decoder or an
# alternative rule can work on this space, so the extractor stays frozen.
#
# Usage:
#   bash run_al_b_feature.sh 3                         # cov-shift ep10, all conds
#   bash run_al_b_feature.sh 3 ep21                     # ep21
#   bash run_al_b_feature.sh 3 ep10 "fog,crosstalk"     # subset

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

echo "=== [B+feature] $TAG [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_b_feature_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "bfeat_$TAG" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_b_feature_$TAG.json" \
  2>&1 | tee "logs/al_b_feature_$TAG.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== B+FEATURE OK ==="
  echo "Check logs/al_b_feature_$TAG.log:"
  echo "  - B1 oracle r17 vs pool r4/r8 vs full-probe (decoder fix?)"
  echo "  - feat: raw vs HDC linear, kNN, prototype (space potential?)"
else
  echo "=== B+FEATURE FAILED (exit $RC) ==="
  exit $RC
fi
