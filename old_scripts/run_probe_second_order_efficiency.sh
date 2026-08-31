#!/usr/bin/env bash
# Cheaper second-order updates: hard-point coreset + dual ridge, MATRIX-FREE CG,
# sparse covariance. Iteration 6 showed the probe gain is cross-coordinate, so the
# question is making the SECOND-ORDER problem cheaper. Eval-only.
#
# CG is now truly matrix-free (Sv = X^T(Xv), no 10k x 10k S) and all timings are
# CUDA-synchronized (real GPU wall time).
#
# Usage:
#   bash run_probe_second_order_efficiency.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_second_order_efficiency.sh 3 "fog" ep10

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

run_so() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [second_order] $label [$CONDS]: cheaper 2nd-order updates ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_second_order_efficiency_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_second_order_${label}.json" \
    2>&1 | tee "logs/probe_second_order_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_so "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_so "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== SECOND-ORDER OK ==="
  echo "Check logs/probe_second_order_{covshift_ep10,covshift_ep21}.log:"
  echo "  coreset: does a small LOW-MARGIN coreset (m=500-2000) + dual solve reach"
  echo "           near the full ridge ceiling (full d-dims, tiny m x m solve)?"
  echo "  cg     : does matrix-free CG converge in few iters (no d^2 S storage)?"
  echo "  sparse : does a small fraction of off-diagonal S recover the gain?"
else
  echo "=== SECOND-ORDER FAILED ==="
  exit 1
fi
