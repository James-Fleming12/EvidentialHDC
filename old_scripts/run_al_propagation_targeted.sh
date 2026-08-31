#!/usr/bin/env bash
# run_al_propagation_targeted.sh: the two NEVER-TESTED levers on the
# propagated-mean decoder (DGLSS++ fog/crosstalk only).
#
# A. TARGETED ANCHOR ACQUISITION (the P3 idea -- frozen-error mass is
#    concentrated in the driveable_surface/sidewalk/terrain pairs):
#    A1 p3_margin  per-class LOWEST-margin anchors (label the frontier,
#                  label-free + deployable) -- the complement of B3 (neg).
#    A4 p3_prior   budget concentrated on classes 11/13/14 at fixed total
#                  (fixed prior, deployable without labels).
#    A2 err_alloc  budget ~ per-class frozen-error mass (oracle allocation
#                  ceiling; needs pool labels).
#    A3 both       A2 allocation + within-class margin selection (ceiling).
#    baselines: gcP (random, current), B2 mass-stratified.
#
# B. COMPOSITION: propagation + feature-conditioned calibration on the SAME
#    labeled set (the anchors). gc_comp vs gcP (adds?), vs gcP + gc_cal_alone
#    (compounds?), vs gc_comp_shuf (noise?).
#
# Decisive:
#   A1/A4 > gcP by >= +0.1  -> acquisition is the lever (deployable)
#   A2/A3 >> gcP but A1/A4 ~ gcP -> the gain needs the label-error oracle
#   gc_comp > gcP + gc_cal_alone -> the mechanisms compound
#   gc_comp ~ gcP            -> calibration does not add to propagation
#   gc_comp ~ gc_comp_shuf   -> the composition gain is noise
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_targeted.sh 3
#   SMOKE=1   bash run_al_propagation_targeted.sh 3
#   bash run_al_propagation_targeted.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_targeted.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_targeted_dglsspp.json
#   logs/al_propagation_targeted_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Targeted acquisition + composition (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --tta_augs 3 --knn 5 --knn_sub 2000 --b_anchors 2"
  echo "  [SMOKE] $SMOKE_ARGS"
fi

FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

IFS=',' read -ra EXS <<< "$EXTRACTORS"
for entry in "${EXS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"
  method="${rest%%|*}"; ckpt="${rest#*|}"
  echo ""
  echo "======================================================"
  echo "=== extractor: $label (method=$method, ckpt=$ckpt) ==="
  echo "======================================================"
  logf="logs/al_propagation_targeted_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_targeted_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_targeted_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    $SMOKE_ARGS --out \"$outjson\""
  echo "  CMD: $CMD"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] not executed"
    continue
  fi
  if eval "$CMD" 2>&1 | tee "$logf"; then
    echo "  [$label] OK -> $outjson"
  else
    echo "  [$label] FAILED -- tail of $logf:"
    tail -25 "$logf"
    fail "$label"
  fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$FAIL" = false ]; then
  echo "=== TARGETED ACQUISITION + COMPOSITION OK ==="
  echo "  A1/A4 > gcP +0.1            -> acquisition is the lever (deployable)"
  echo "  A2/A3 >> gcP but A1/A4 ~ gcP -> needs the label-error oracle"
  echo "  gc_comp > gcP + gc_cal_alone -> the mechanisms compound"
else
  echo "=== TARGETED ACQUISITION + COMPOSITION FAILED ==="
  exit 1
fi
