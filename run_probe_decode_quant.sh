#!/usr/bin/env bash
# run_probe_decode_quant.sh: probe-efficiency diagnostic (inference side).
# Decode speed + ceiling for the R4 probe: fp32 (current) vs int8-quantized W
# vs +-1 W (old block-sign integer form) vs the low-rank factored decode
# W = W0 + U8 C (delta matmul 10k x 8 instead of 10k x 17).
#
# Usage:
#   bash run_probe_decode_quant.sh 1
#   CONDS="fog,wet_ground" bash run_probe_decode_quant.sh 1
#
# Output: robust_diagnostic/logs/probe_decode_quant_ep10.json

set -u
set -o pipefail
GPU="${1:-1}"
CONDS="${CONDS:-fog,wet_ground}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [decode-quant] $CONDS on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_decode_quant_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "decode_quant_ep10" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/probe_decode_quant_ep10.json" \
  2>&1 | tee "logs/probe_decode_quant_ep10.log"
RC=$?
[ $RC -eq 0 ] && echo "=== DECODE-QUANT OK ===" || { echo "FAILED ($RC)"; exit $RC; }
