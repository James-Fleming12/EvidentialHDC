#!/usr/bin/env bash
# al_query_rule: the Iteration-1 query-rule comparison for the one-label-per-
# cluster AL framework (oracle-simulated). Four rules, budget -> mIoU curves,
# plus efficiency, for the README table:
#   influence       : rank clusters by J_c = sum of per-point influence I_i
#                     (the exact magnitude of the point's W contribution)
#   confidence      : rank by representative confidence, ascending (uncertainty
#                     sampling, the free baseline)
#   influence_gated : influence with the prototype-vs-probe disagreement gate
#                     (only disagreeing clusters are eligible)
#   confidence_gated: confidence with the same gate
# References: frozen / oracle (all pool labeled) / grounded_all (the grounding
# scheme's own ceiling). Efficiency: per-budget ridge fit time + kmeans time.
#
# Usage:
#   bash run_al_query_rule.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_query_rule.sh 3 "fog" ep10

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

run_rule() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_query_rule] $label [$CONDS]: query-rule comparison (agreement-gated grounding) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_query_rule_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_query_rule_${label}.json" \
    2>&1 | tee "logs/al_query_rule_${label}.log" || fail "$label"
  echo ""
  echo "=== [al_query_rule] $label [$CONDS]: DISTANCE-ONLY grounding control ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_query_rule_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "${label}_nodistgate" --conds "$CONDS" \
    --ground_agreement 0 \
    --out "robust_diagnostic/logs/al_query_rule_${label}_nodistgate.json" \
    2>&1 | tee "logs/al_query_rule_${label}_nodistgate.log" || fail "${label}_nodistgate"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_rule "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_rule "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-QUERY-RULE OK ==="
  echo "Check logs/al_query_rule_{covshift_ep10,covshift_ep21}.log (agreement-gated)"
  echo "and the *_nodistgate variants (distance-only grounding control): per-K"
  echo "budget->mIoU for the four rules, grounded_all ceiling, efficiency."
  echo "Compare the two: the agreement gate should lift grounded_all above frozen"
  echo "(distance-only poisons T with ~35% wrong propagated labels)."
else
  echo "=== AL-QUERY-RULE FAILED ==="
  exit 1
fi
