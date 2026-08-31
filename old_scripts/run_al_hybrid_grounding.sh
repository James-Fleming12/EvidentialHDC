#!/usr/bin/env bash
# al_hybrid_grounding: the Iteration-5 hybrid AL mechanism test. Compounds the
# three measured geometric promises: spatial superpixel grounding (CCL on the
# projection mask), class-centroid decode from the expanded pools, and an
# agreement gate on T. The ABLATION LADDER per budget (S0 direct / S1 spatial /
# S2 centroid / S3 hybrid-AND / S4 union) shows which stage earns its keep.
# Efficiency measured per stage (CCL, centroid decode, ridge fit).
#
# Usage:
#   bash run_al_hybrid_grounding.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_hybrid_grounding.sh 3 "fog" ep10

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

run_hybrid() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_hybrid_grounding] $label [$CONDS]: superpixel + centroid + gate ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_hybrid_grounding_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_hybrid_grounding_${label}.json" \
    2>&1 | tee "logs/al_hybrid_grounding_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_hybrid "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_hybrid "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-HYBRID-GROUNDING OK ==="
  echo "Check logs/al_hybrid_grounding_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: superpixel component stats, per-budget ablation ladder"
  echo "(S0/S1/S2/S3/S4: prec, cov, miou), CCL + fit times."
else
  echo "=== AL-HYBRID-GROUNDING FAILED ==="
  exit 1
fi
