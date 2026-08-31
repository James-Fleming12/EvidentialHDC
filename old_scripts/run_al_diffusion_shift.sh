#!/usr/bin/env bash
# al_diffusion_shift: the Iteration-2 AL mechanism test. Two replacements for
# the failed cluster+hard-propagation grounding:
#   A. Graph diffusion with QUERIED anchors (influence/confidence/random,
#      class-floored), no clustering, no kNN -- budget -> mIoU curves.
#   B. Partial shift structure: global shift from k labeled classes carried to
#      ALL classes (label-skipping) vs per-class-only vs oracle shift.
# Efficiency: the whole AL loop (diffusion + ridge fit) is ~0.2s, no clustering.
#
# Usage:
#   bash run_al_diffusion_shift.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_diffusion_shift.sh 3 "fog" ep10

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

run_ds() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_diffusion_shift] $label [$CONDS]: diffusion + shift structure ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_diffusion_shift_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_diffusion_shift_${label}.json" \
    2>&1 | tee "logs/al_diffusion_shift_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_ds "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_ds "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-DIFFUSION-SHIFT OK ==="
  echo "Check logs/al_diffusion_shift_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: diffusion budget->mIoU per anchor rule (influence_floor /"
  echo "influence / confidence / random), shift carry_over vs per_class_only vs"
  echo "oracle_shift, efficiency (diffuse + fit, no clustering)."
else
  echo "=== AL-DIFFUSION-SHIFT FAILED ==="
  exit 1
fi
