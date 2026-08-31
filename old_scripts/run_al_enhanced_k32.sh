#!/usr/bin/env bash
# run_al_enhanced_k32.sh: get the most out of k=32 on COV-SHIFT ep10.
# Covers the enhancers you asked to add, all at k=32 (224 labels) where B1
# was already positive: proto, S^{-1}T r=4,8, and B1 r=4,8,17.
# Eval-only, ~1 min per condition.
#
# Usage:
#   bash run_al_enhanced_k32.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [enhanced k32] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_enhanced_k32_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "enhanced_k32_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_enhanced_k32_ep10.json" \
  2>&1 | tee "logs/al_enhanced_k32_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== ENHANCED K32 OK ==="
  echo "Check logs/al_enhanced_k32_ep10.log:"
  echo "  - proto k8 vs k32: does prototype close gap at higher k?"
  echo "  - S^-1T r4/r8 at k=32: does whitened T beat pool-cov?"
  echo "  - B1 r4/r8/r17 at k=32: does r=17 beat r=8? Full low-rank ceiling."
else
  echo "=== ENHANCED K32 FAILED (exit $RC) ==="
  exit $RC
fi
