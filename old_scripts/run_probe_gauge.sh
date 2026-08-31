#!/usr/bin/env bash
# Prototype + separability gauge: does a tiny k-dim probe's advantage over the
# prototype (delta_gauge) predict the full-probe gain? If yes, the probe becomes an
# EXCEPTION HANDLER -- the prototype stays the cheap O(Cd) primary decoder, and the
# expensive Nystrom/ridge correction is only paid when the gauge says it is worth it.
# Also measures the rank-k correction W = mu + V A (cheap boundary rotation).
#
# Usage:
#   bash run_probe_gauge.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_gauge.sh 3 "fog" ep10

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

run_gauge() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [gauge] $label [$CONDS]: prototype + separability gauge ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_gauge_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_gauge_${label}.json" \
    2>&1 | tee "logs/probe_gauge_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_gauge "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_gauge "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== GAUGE OK ==="
  echo "Check logs/probe_gauge_{covshift_ep10,covshift_ep21}.log:"
  echo "  delta_gauge (tiny k-dim probe - prototype) vs full_gain (R4 - proto):"
  echo "    if corr is high, the gauge gates the expensive refit -- prototype stays"
  echo "    the cheap default, probe is the exception handler."
  echo "  rank-k corr: how much of the R4 ceiling the cheap W=mu+VA correction recovers."
else
  echo "=== GAUGE FAILED ==="
  exit 1
fi
