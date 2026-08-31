#!/usr/bin/env bash
# run_al_rule_budget.sh: the two AL tests on the 21ep ball/spec spaces.
#   TEST1 (query rules): influence vs confidence vs random vs centroid-near,
#     k=8, oracle counts, V3 control variate (rho=0.5), with per-rule mean
#     quality so a rule's failure is attributable to selection vs premise.
#   TEST2 (k budget): k in {2,4,8} x rho in {0.25,0.5,0.75}, best rule from
#     TEST1, source counts (deployable). Does k=2 halve the label cost?
#   PREMISE: closeable gap, t_cos/w_cos chain, whitened error at the best
#     config -- the guardrail for "is the space still worth it".
# Eval-only, ~1 min per condition.
#
# Usage:
#   bash run_al_rule_budget.sh 3                         # ball+spec, final ckpt
#   bash run_al_rule_budget.sh 3 valid_best              # best-val ckpt
#   bash run_al_rule_budget.sh 3 final "fog,crosstalk"   # subset

set -u
set -o pipefail
GPU="${1:-3}"
TAG="${2:-final}"
CONDS="${3:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, tag=$TAG, conds=$CONDS"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_one() {
  local method="$1"; local label="$2"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  if [ "$TAG" = "valid_best" ]; then
    ckpt_dir="$ckpt_dir/valid_best"
    if [ ! -f "$ckpt_dir/SENet" ]; then
      echo "=== [$label:$TAG] SKIP: no SENet copy ==="
      return 0
    fi
  fi
  echo ""
  echo "=== [rule-budget] $label:$TAG [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_rule_budget_diag.py \
    --path_b "$ckpt_dir" --method_b "$method" --label "${label}_${TAG}" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_rule_budget_${label}_${TAG}.json" \
    2>&1 | tee "logs/al_rule_budget_${label}_${TAG}.log" || fail "$label:$TAG"
}

run_one "$BALL" "ball"
run_one "$SPEC" "spec"

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-RULE-BUDGET OK ==="
  echo "Check logs/al_rule_budget_{ball,spec}_${TAG}.log:"
  echo "  TEST1: confidence vs influence delta + mean quality per rule"
  echo "  TEST2: k=2 vs k=8 across rho (does the budget halve?)"
  echo "  PREMISE: closeable gap / t_cos / w_cos / whitened error at best cfg"
else
  echo "=== AL-RULE-BUDGET FAILED ==="
  exit 1
fi
