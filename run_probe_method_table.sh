#!/usr/bin/env bash
# README method table: zero-shot + ceiling + efficiency for R1 prototype / full
# probe (R4) / block_ridge float / block_ridge sign (the candidate). Eval-only.
#
# Usage:
#   bash run_probe_method_table.sh 3                # ep10+ep21, all 4 conditions
#   bash run_probe_method_table.sh 3 "snow,wet_ground,fog,crosstalk" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-snow,wet_ground,fog,crosstalk}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_table() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [method_table] $label [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_method_table_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_method_table_${label}.json" \
    2>&1 | tee "logs/probe_method_table_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_table "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_table "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== METHOD-TABLE OK ==="
  echo "Check logs/probe_method_table_{covshift_ep10,covshift_ep21}.log:"
  echo "  block_ridge sign (*) is the candidate: HDC-native block-diagonal ridge at"
  echo "  10000-d with quantized +-1 W (integer popcount decode)."
  echo "  Compare its zero-shot/ceiling vs R1 prototype (baseline) and full probe (R4),"
  echo "  and update/decode pts/s for the README table."
else
  echo "=== METHOD-TABLE FAILED ==="
  exit 1
fi
