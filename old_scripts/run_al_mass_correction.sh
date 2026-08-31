#!/usr/bin/env bash
# al_mass_correction: Iteration-8 -- the mass-calibration and mean-estimation
# routes. SECTION 1 (the deciding experiment, run first): 8D oracle-count
# ceiling -- if it reaches the oracle, the problem is class-mass calibration.
# SECTION 2: the mass-correction family (8A raw / 8E normalized / 8F
# source-count prior / 8G top-K with corrected counts). SECTION 3: the
# mean-estimation route (bulk sampling strategies, source-count + target-mean
# synthesis, control-variate shrinkage with the ridge-relevant whitened error).
#
# Usage:
#   bash run_al_mass_correction.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_mass_correction.sh 3 "fog" ep10

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

run_mc() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_mass_correction] $label [$CONDS]: mass calibration + mean route ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_mass_correction_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_mass_correction_${label}.json" \
    2>&1 | tee "logs/al_mass_correction_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_mc "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_mc "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-MASS-CORRECTION OK ==="
  echo "Check logs/al_mass_correction_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: 8D verdict (STRONG/PARTIAL/WEAK), 8A/8E/8F/8G miou,"
  echo "mean-estimator comparison, 3b source-count synthesis, 3c shrinkage"
  echo "with the whitened error."
else
  echo "=== AL-MASS-CORRECTION FAILED ==="
  exit 1
fi
