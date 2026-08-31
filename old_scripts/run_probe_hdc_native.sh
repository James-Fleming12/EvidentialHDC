#!/usr/bin/env bash
# HDC-native efficiency scan for the linear probe: binary-space classification
# (quantized W -> integer/popcount decode) and binary-space updates (integer Gram,
# block-diagonal ridge, sample-space RLS streaming). Keeps the 10000-d HDC code.
#
# Usage:
#   bash run_probe_hdc_native.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_hdc_native.sh 3 "fog" ep10

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

run_native() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [hdc_native] $label [$CONDS]: binary-space classification + updates ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_hdc_native_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_hdc_native_${label}.json" \
    2>&1 | tee "logs/probe_hdc_native_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_native "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_native "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== HDC-NATIVE OK ==="
  echo "Check logs/probe_hdc_native_{covshift_ep10,covshift_ep21}.log:"
  echo "  decode_sign vs decode_float : does quantizing W to +-1 keep mIoU (integer"
  echo "                                decode = d - 2*Hamming on packed bits)?"
  echo "  dual_int vs dual_float      : exact integer Gram (no float rounding)."
  echo "  block_ridge                 : B small (d/B)^3 solves vs one d^3 -- the win."
  echo "  dual_rls                    : sample-space streaming, O(n^2)/point."
else
  echo "=== HDC-NATIVE FAILED ==="
  exit 1
fi
