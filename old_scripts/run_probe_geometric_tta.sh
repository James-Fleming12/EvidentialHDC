#!/usr/bin/env bash
# probe_geometric_tta: label-free GEOMETRIC test-time adaptation for the probe,
# abandoning T entirely (Iteration-11 finding: the pseudo-label half is poisoned).
#   A. Subspace alignment / Procrustes (plain basis match + t2c/c2t rotations,
#      k in {8,32,128,1000}) -- exact if the corruption is a rotation.
#   B. CORAL covariance alignment (S_t^-1/2 S_c^1/2 W_zs, rank {128,256,1000})
#      + whitening-only control.
#   C. Label diffusion on the point graph (top-1%/5% anchors, a in {0.1,0.5,0.9})
#      + oracle-anchored upper bound.
#   Controls: mean-shift decode-time bias. Everything matrix-free on X.
#
# Usage:
#   bash run_probe_geometric_tta.sh 3                  # ep10+ep21, all 4 conds
#   bash run_probe_geometric_tta.sh 3 "fog" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_geo() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [geometric_tta] $label [$CONDS]: S-only label-free adaptation ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_geometric_tta_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_geometric_tta_${label}.json" \
    2>&1 | tee "logs/probe_geometric_tta_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_geo "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_geo "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== GEOMETRIC-TTA OK ==="
  echo "Check logs/probe_geometric_tta_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json. spectral_overlap says whether the corruption is a pure"
  echo "rotation (all ~1 -> procrustes k*_plain should be exact); procrustes/"
  echo "coral/diffusion give the S-only mIoU vs frozen/oracle."
else
  echo "=== GEOMETRIC-TTA FAILED ==="
  exit 1
fi
