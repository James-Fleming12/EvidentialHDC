#!/usr/bin/env bash
# Iteration-0 probe-gap diagnostic: characterize what TTA/AL must improve on between
# the zero-shot (frozen clean-fit) and labeled (pool-refit) linear-probe decoders.
# Runs on the cov-shift ep10/ep21 weights, eval-only.
#
# Per condition (snow/wet_ground/fog/crosstalk) it reports:
#   - shift type        (cos(W_zs, W_oracle): pure translation vs rotation)
#   - bias-only share   (how much of the gap a gradient-free intercept re-center closes)
#   - margin            (frozen-probe margin of correct/wrong/oracle-fixed points)
#   - outlier/norm      (norm of correct/wrong/oracle-fixed points)
#   - per-class gains   (the TTA/AL target classes)
#   - pool-size curve   (oracle mIoU vs 1k/10k/50k/100k labeled pool = the AL budget)
#
# Usage:
#   bash run_probe_gap_diag.sh 3            # GPU 3, ep10 + ep21
#   bash run_probe_gap_diag.sh 3 ep10       # only ep10

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

run_gap() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [probe_gap] $label: zero-shot vs labeled probe gap ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_gap_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/probe_gap_$label.json" \
    > "logs/probe_gap_$label.log" 2>&1 || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_gap "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_gap "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== PROBE-GAP OK ==="
  echo "Check logs/probe_gap_{covshift_ep10,covshift_ep21}.log:"
  echo "  shift.mean_cos_W      : ~1 = translation (bias-only enough); <<1 = rotation"
  echo "  bias_only.share_of_gap: the gradient-free intercept-recenter share"
  echo "  margin/norm           : what gates could and couldn't fix"
  echo "  per_class + pool_curve: the AL target classes and label budget"
else
  echo "=== PROBE-GAP FAILED ==="
  exit 1
fi
