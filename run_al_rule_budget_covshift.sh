#!/usr/bin/env bash
# run_al_rule_budget_covshift.sh: the C16 AL recipe (centroid k=2 + source
# counts + control variate + fractional-residual) on the COV-SHIFT extractor.
# The C16 tests were on the corsupcon ball/spec spaces, which have LOWER
# fog/crosstalk ceilings than the cov-shift (inputin_in_chan) extractor (fog
# 0.252 vs 0.433, crosstalk 0.401 vs 0.594). The open question: does the same
# cheap AL recipe reach the cov-shift HIGH ceilings on fog/crosstalk?
#
# Runs the identical al_rule_budget_diag on both the ep10 (optimal window)
# and ep21 cov-shift checkpoints.
#
# Usage:
#   bash run_al_rule_budget_covshift.sh 3                  # ep10+ep21
#   bash run_al_rule_budget_covshift.sh 3 ep10             # ep10 only
#   bash run_al_rule_budget_covshift.sh 3 all "fog,crosstalk"

set -u
set -o pipefail
GPU="${1:-3}"
ONLY="${2:-all}"
CONDS="${3:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, only=$ONLY, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_one() {
  local ckpt="$1"; local tag="$2"
  echo ""
  echo "=== [rule-budget covshift:$tag] $CONDS ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_rule_budget_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "covshift_${tag}" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_rule_budget_covshift_${tag}.json" \
    2>&1 | tee "logs/al_rule_budget_covshift_${tag}.log" || fail "covshift:$tag"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_one "$EP10_CKPT" "ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_one "$EP21_CKPT" "ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-RULE-BUDGET-COVSHIFT OK ==="
  echo "Compare vs the ball/spec C16 numbers (al_rule_budget_{ball,spec}_final.json):"
  echo "  - does the centroid k=2 recipe reach the cov-shift HIGH ceilings on"
  echo "    fog/crosstalk (0.43/0.59) vs ball/spec's +0.05..0.08 on lower ones?"
  echo "  - do the snow/wet healthy-condition results hold (the ball/spec frozen"
  echo "    was near its own ceiling; cov-shift frozen is lower, so the closeable"
  echo "    gap is bigger there -- does AL now close it?)"
else
  echo "=== AL-RULE-BUDGET-COVSHIFT FAILED ==="
  exit 1
fi
