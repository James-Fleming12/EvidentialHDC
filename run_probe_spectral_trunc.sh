#!/usr/bin/env bash
# run_probe_spectral_trunc.sh: probe-efficiency diagnostic (training side).
# Truncated spectral factorization (randomized top-K of S = X^T X, matrix-free)
# vs the full eigh/solve (accurate, ~4s) vs Nystrom-warm CG-8/20 (fast,
# under-converges). Target: exact-solve accuracy at CG-class cost.
#
# Usage:
#   bash run_probe_spectral_trunc.sh 1
#   CONDS="wet_ground,fog" bash run_probe_spectral_trunc.sh 1
#
# Output: robust_diagnostic/logs/probe_spectral_trunc_ep10.json

set -u
set -o pipefail
GPU="${1:-1}"
CONDS="${CONDS:-wet_ground,fog}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [spectral-trunc] $CONDS on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_spectral_trunc_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "spec_trunc_ep10" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/probe_spectral_trunc_ep10.json" \
  2>&1 | tee "logs/probe_spectral_trunc_ep10.log"
RC=$?
[ $RC -eq 0 ] && echo "=== SPECTRAL-TRUNC OK ===" || { echo "FAILED ($RC)"; exit $RC; }
