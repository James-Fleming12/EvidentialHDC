#!/usr/bin/env bash
# run_al_c29.sh: C29 comprehensive breadth-first test (your points 1-36, bets 1-3).
# Keeps COV-SHIFT frozen (no extractor change) and tests the three hypothesis
# families before committing to one. Eval-only, ~2 min/condition on one GPU.
#
# C29A bank information: random, confidence-stratified, boundary-heavy, mixed 50/50
# C29B bank-derived correction: G = X_B^T (Y_B - P0) with 4 weightings, W = W0 + eta*G and W = W0 + U_G*C
# C29C consensus U: 500 -> 5x100 groups, SVD each G_b, consensus subspace, r=1,2,4,8
# C30/C31 stabilization + budget: ridge, clipping, fractional, trust-region + q extra labels
#
# Usage:
#   bash run_al_c29.sh 3                         # cov-shift ep10, all conds
#   bash run_al_c29.sh 3 ep21                     # ep21
#   bash run_al_c29.sh 3 ep10 "fog,crosstalk"     # subset

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

echo "=== [C29] $TAG [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_c29_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "c29_$TAG" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_c29_$TAG.json" \
  2>&1 | tee "logs/al_c29_$TAG.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== C29 OK ==="
  echo "Check logs/al_c29_$TAG.log:"
  echo "  - Bet 1 (bank->G): which bank gives best G quality and W0+G delta"
  echo "  - Bet 2 (consensus U): does stable subspace beat single-group U"
  echo "  - Bet 3 (boundary vs representative): 1-NN vs W correction"
else
  echo "=== C29 FAILED (exit $RC) ==="
  exit $RC
fi
