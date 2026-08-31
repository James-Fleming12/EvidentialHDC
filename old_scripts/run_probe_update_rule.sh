#!/usr/bin/env bash
# Validate the gradient-free probe UPDATE RULES (option 2 ridge, option 3 FLDA)
# against the LR oracle and the R1 prototype, on the cov-shift ep10/ep21 weights.
# Eval-only: correctness (reaches oracle?), efficiency (accumulate+solve wall-clock,
# backprop-free), equivalence (accumulate == batch).
#
# Usage:
#   bash run_probe_update_rule.sh 3            # GPU 3, ep10 + ep21
#   bash run_probe_update_rule.sh 3 ep10       # only ep10

set -u
set -o pipefail
GPU="${1:-3}"
ONLY="${2:-all}"
echo "Using GPU $GPU"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_rule() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [update_rule] $label: ridge/FLDA vs LR oracle vs prototype ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_update_rule_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/probe_update_rule_$label.json" \
    2>&1 | tee "logs/probe_update_rule_$label.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_rule "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_rule "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== UPDATE-RULE OK ==="
  echo "Check logs/probe_update_rule_{covshift_ep10,covshift_ep21}.log:"
  echo "  Ridge-accum vs LR oracle : does the gradient-free rule reach the ceiling?"
  echo "  accum==batch max|W diff| : is the accumulate-and-solve exactly the batch form?"
  echo "  accumulate+solve wall    : is it prototype-cheap (backprop-free, additions+solve)?"
  echo "  FLDA                     : does option 3 reach the oracle too?"
else
  echo "=== UPDATE-RULE FAILED ==="
  exit 1
fi
