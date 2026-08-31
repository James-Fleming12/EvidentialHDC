#!/usr/bin/env bash
# run_al_comprehensive.sh: the 2-question check on the 21ep ball/spec feature spaces.
#   Q1: does a CHEAP tweak of the previous AL method work on the new spaces?
#   Q2: what other properties of these spaces enable a cheap/efficient AL framework?
# Eval-only, ~15-20 min per extractor on one GPU.
#
# Usage:
#   bash run_al_comprehensive.sh 3                         # ball+spec, final ckpt
#   bash run_al_comprehensive.sh 3 valid_best              # gate the best-val ckpt
#   bash run_al_comprehensive.sh 3 final "fog,crosstalk"   # subset

set -u
set -o pipefail
GPU="${1:-3}"
TAG="${2:-final}"
CONDS="${3:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, tag=$TAG, conds=$CONDS"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_one() {
  local method="$1"; local label="$2"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  if [ "$TAG" = "valid_best" ]; then
    ckpt_dir="$ckpt_dir/valid_best"
    if [ ! -f "$ckpt_dir/SENet" ]; then
      echo "=== [$label:$TAG] SKIP: no SENet copy ==="
      return 0
    fi
  fi
  echo ""
  echo "=== [comprehensive] $label:$TAG [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_comprehensive_diag.py \
    --path_b "$ckpt_dir" --method_b "$method" --label "${label}_${TAG}" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_comprehensive_${label}_${TAG}.json" \
    2>&1 | tee "logs/al_comprehensive_${label}_${TAG}.log" || fail "$label:$TAG"
}

run_one "$BALL" "ball"
run_one "$SPEC" "spec"

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-COMPREHENSIVE OK ==="
  echo "Check logs/al_comprehensive_{ball,spec}_${TAG}.log:"
  echo "  - slight variations: is any V1-V4 positive on snow/wet or > V0?"
  echo "  - properties: R1 vs linear, kappa/prank, mean-k, leverage, per-class"
else
  echo "=== AL-COMPREHENSIVE FAILED ==="
  exit 1
fi
