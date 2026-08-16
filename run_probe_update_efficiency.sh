#!/usr/bin/env bash
# Accuracy x efficiency table for the ridge update vs the R1 prototype pipeline.
# Runs primal / dual (Woodbury) / RLS (Sherman-Morrison) over pool sizes, with the
# R1 prototype fit+update+decode as the throughput reference (it/s). Eval-only.
#
# Conditions default to wet_ground,fog (the largest-gap and rotation-heavy cases);
# pass --conds via the script's 2nd arg (comma-separated).
#
# Usage:
#   bash run_probe_update_efficiency.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_update_efficiency.sh 3 "fog,crosstalk" ep10
#   NUSC_DIR not needed; conds are SemanticKITTI-C.

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

run_eff() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [efficiency] $label [$CONDS]: probe forms vs R1 prototype ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_update_efficiency_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_update_eff_${label}.json" \
    2>&1 | tee "logs/probe_update_eff_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_eff "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_eff "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== EFFICIENCY OK ==="
  echo "Check logs/probe_update_eff_{covshift_ep10,covshift_ep21}.log:"
  echo "  mIoU across primal/dual/RLS at the same pool: must MATCH (same W)."
  echo "  pts/s vs the R1 prototype fit/update/decode: the probe's throughput overhead."
  echo "  wall-clock crossover: dual wins at small n, RLS for streaming, primal baseline."
else
  echo "=== EFFICIENCY FAILED ==="
  exit 1
fi
