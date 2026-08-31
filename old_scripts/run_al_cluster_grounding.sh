#!/usr/bin/env bash
# al_cluster_grounding: AL-framework preliminary diagnostics on the CURRENT
# cov-shift setup. Verifies the dense per-class cluster packing (NN purity,
# k-means purity at K=#classes, separation) and measures how few labels the
# space needs:
#   A. packing (pool vs clean reference, per class)
#   B. one-label-per-cluster grounding: budget(K) -> coverage curve +
#      distance-gated coverage (the "label if close, else ask" rule)
#   C. label-reduction properties: per-class shift alignment (global transform
#      carry-over), confidence-representativeness, within-class multi-modality,
#      pseudo-label agreement vs distance to centroid.
#
# Usage:
#   bash run_al_cluster_grounding.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_cluster_grounding.sh 3 "fog" ep10

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

run_ground() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_cluster_grounding] $label [$CONDS]: AL packing + label budget ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_cluster_grounding_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_cluster_grounding_${label}.json" \
    2>&1 | tee "logs/al_cluster_grounding_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_ground "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_ground "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-CLUSTER-GROUNDING OK ==="
  echo "Check logs/al_cluster_grounding_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: packing (nn1/nnk, cluster purity), grounding budget->coverage,"
  echo "distance-gated coverage, shift alignment, confidence-representativeness,"
  echo "within-class multi-modality."
else
  echo "=== AL-CLUSTER-GROUNDING FAILED ==="
  exit 1
fi
