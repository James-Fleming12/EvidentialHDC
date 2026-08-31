#!/usr/bin/env bash
# run_al_prototype_sinv.sh: prototypes + S^{-1}T-derived U + B1 budget (no memory banks).
# All eval-only on COV-SHIFT ep10, 4 conditions. Keeps extractor frozen.
#
# Usage:
#   bash run_al_prototype_sinv.sh 3                         # ep10, all conds

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [proto+sinv+B1] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_prototype_sinv_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "protosinv_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_prototype_sinv_ep10.json" \
  2>&1 | tee "logs/al_prototype_sinv_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== PROTO+SINV+B1 OK ==="
  echo "Check logs/al_prototype_sinv_ep10.log:"
  echo "  - proto k=8 vs k=32/64/128: does prototype close the gap at higher k?"
  echo "  - S^-1T r4/r8 vs pool-cov C22 ~0: does whitened T basis beat it?"
  echo "  - B1 k=8..128: does raising budget help current linear B1?"
else
  echo "=== PROTO+SINV+B1 FAILED (exit $RC) ==="
  exit $RC
fi
