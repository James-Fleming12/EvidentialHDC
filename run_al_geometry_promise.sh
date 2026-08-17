#!/usr/bin/env bash
# al_geometry_promise: the intrinsic information content of the feature space's
# geometric properties for ultra-cheap labeling, measured in the 128-d space
# (where the packing actually lives) with robustness (mean+-std over repeated
# random anchor draws):
#   A. nearest-anchor cosine gate (1 queried point/class)
#   B. class-centroid cosine (1 anchor/class) + ORACLE centroid ceiling
#   C. multi-anchor MIN-cosine agreement (2 anchors/class)
#   D. spatial adjacency promise (label grids: P(same class | projection
#      neighbor), per class) -- the superpixel grounding
#   E. per-class 128-d NN purity (budget allocation)
#   F. confidence-conditioned packing (frozen probe confidence x geometry)
#
# Usage:
#   bash run_al_geometry_promise.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_geometry_promise.sh 3 "fog" ep10

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

run_promise() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_geometry_promise] $label [$CONDS]: 128-d geometry promises ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_promise_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_geometry_promise_${label}.json" \
    2>&1 | tee "logs/al_geometry_promise_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_promise "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_promise "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-GEOMETRY-PROMISE OK ==="
  echo "Check logs/al_geometry_promise_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: A/B/C precision@coverage (mean+-std over random draws),"
  echo "spatial P4/P8 + per-class coherence, per-class NN purity, F confidence"
  echo "conditioning."
else
  echo "=== AL-GEOMETRY-PROMISE FAILED ==="
  exit 1
fi
