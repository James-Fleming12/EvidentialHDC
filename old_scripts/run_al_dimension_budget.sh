#!/usr/bin/env bash
# al_dimension_budget: the Iteration-6 dimension test. Does a smaller code space
# make labels cheaper? The S0/direct-label budget curve (influence-ranked,
# class-floored, NO expansion) across code dims {128 real-valued, 512, 1k, 2k,
# 5k, 10k binarized}: per dim, budget -> mIoU + frozen/oracle refs + the
# crossing and 90%-approach budgets. Tests the estimation argument
# (labels-per-class ~ d) and re-checks the C8 ceiling-invariance claim.
#
# Usage:
#   bash run_al_dimension_budget.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_dimension_budget.sh 3 "fog" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_dim() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_dimension_budget] $label [$CONDS]: S0 budget curve across dims ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_dimension_budget_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_dimension_budget_${label}.json" \
    2>&1 | tee "logs/al_dimension_budget_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_dim "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_dim "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-DIMENSION-BUDGET OK ==="
  echo "Check logs/al_dimension_budget_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: per-dim budget->mIoU, frozen/oracle refs, cross_budget,"
  echo "approach_budget. If smaller dims cross/approach at 10-100x smaller"
  echo "budgets, the dim is the path-cheapening lever for the AL story."
else
  echo "=== AL-DIMENSION-BUDGET FAILED ==="
  exit 1
fi
