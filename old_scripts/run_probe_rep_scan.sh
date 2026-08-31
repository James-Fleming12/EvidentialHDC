#!/usr/bin/env bash
# Efficiency-lever scan for the linear probe, HDC-native FIRST.
#
# The efficiency win comes from USING the binarization (Section A), not from
# shrinking the projection. Section A stays at the 10000-d HDC code:
#   - diag_ridge: for +/-1 codes diag(X^T X)=n, so diagonal ridge == the prototype
#     (proves the probe's gain is the off-diagonal covariance; the HDC-free bound).
#   - dual_woodbury: the HDC-native pooled update (inversion in the sample dim n,
#     integer +/-1 dot-product matrix G = X X^T).
#   - rls: Sherman-Morrison streaming, no solve.
# Section B is the DIMENSION CHECK (a paper claim about the PROJECTION, not the
# method): is the probe's gain a property of the large 10000-d projection, or of the
# BINARIZED GEOMETRY? If mIoU holds at small k/d', the projection size never helped
# -- we keep 10000-d + binarization; the check just shows the size was never the
# source of the power.
#
# Usage:
#   bash run_probe_rep_scan.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_rep_scan.sh 3 "fog" ep10

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

run_scan() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [rep_scan] $label [$CONDS]: HDC-native levers + projection dimension check ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_rep_scan_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_rep_scan_${label}.json" \
    2>&1 | tee "logs/probe_rep_scan_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_scan "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_scan "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== REP-SCAN OK ==="
  echo "Check logs/probe_rep_scan_{covshift_ep10,covshift_ep21}.log:"
  echo "  Section A (HDC-native, the method): diag_ridge==proto proves the gain is"
  echo "    covariance; dual_woodbury / rls are the efficient HDC forms."
  echo "  Section B (dimension check, the projection claim): if jl_k/code_d keep mIoU"
  echo "    at small k/d', the projection SIZE never helped -- binarized geometry did."
else
  echo "=== REP-SCAN FAILED ==="
  exit 1
fi
