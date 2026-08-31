#!/usr/bin/env bash
# Iteration-9 follow-up: avoid the hard pseudo-label gate (which starves the
# covariance) via (A) confidence-weighted updates and (B) two-stage updates.
#
#   A weighted (w=conf / conf^2 / margin): wrong points contribute weakly, all
#     points' covariance kept -- soft weighting, no admit/veto.
#   B two-stage: fit on frozen pseudo-labels, re-gate / reweight by the UPDATED
#     probe's confidence (hard re-gate, soft weight, soft-then-hard).
#
# References: frozen (no update), oracle (true labels), no_gate (all pseudo-labels).
#
# Usage:
#   bash run_probe_weighted_two_stage.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_weighted_two_stage.sh 3 "fog" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-wet_ground,fog}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_ws() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [weighted_2stage] $label [$CONDS]: weighted + two-stage pseudo-label update ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_weighted_two_stage_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_weighted_2stage_${label}.json" \
    2>&1 | tee "logs/probe_weighted_2stage_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_ws "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_ws "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== WEIGHTED-2STAGE OK ==="
  echo "Check logs/probe_weighted_2stage_{covshift_ep10,covshift_ep21}.log:"
  echo "  A weighted: does soft weighting (w=conf/conf^2/margin) climb from no_gate"
  echo "             toward oracle by keeping all points' covariance?"
  echo "  B two-stage: does the UPDATED probe's confidence (re-gate / reweight) do"
  echo "             better than Iteration 9's failed first-round gate?"
else
  echo "=== WEIGHTED-2STAGE FAILED ==="
  exit 1
fi
