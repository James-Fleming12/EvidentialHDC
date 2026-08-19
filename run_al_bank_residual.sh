#!/usr/bin/env bash
# run_al_bank_residual.sh: old tiny bank (500) on the new stable residual update.
# Tests whether the 4 old bank allocations immediately improve with W = W0 + U_r C
# (r=8, oracle U, eta=1) instead of the old full-probe W_pseudo.
# Eval-only on COV-SHIFT ep10, 4 conditions.
#
# Usage:
#   bash run_al_bank_residual.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [bank+residual] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_bank_residual_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "bank_residual_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_bank_residual_ep10.json" \
  2>&1 | tee "logs/al_bank_residual_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== BANK+RESIDUAL OK ==="
  echo "Check logs/al_bank_residual_ep10.log:"
  echo "  - bank mIoU (random/uniform/diverse/uncertainty) vs frozen"
  echo "  - W_pseudo (old) vs W_res_pseudo (new) delta at 56+500"
else
  echo "=== BANK+RESIDUAL FAILED (exit $RC) ==="
  exit $RC
fi
