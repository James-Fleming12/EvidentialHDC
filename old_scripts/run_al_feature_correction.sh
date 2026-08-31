#!/usr/bin/env bash
# run_al_feature_correction.sh: the 8B branch -- is the oracle decision
# correction delta_z* predictable from a RICH label-free feature set, and can a
# few labels calibrate it? (DGLSS++ only, fog/crosstalk)
#
# P1 ORACLE CEILING (rich features): mean R^2, gc_rich (linear), gc_rff
#    (nonlinear), feature ablation (which features carry the predictability).
# P2 FEW-LABEL CALIBRATION: fit the same model on b labels/class (target = the
#    label residual Y - softmax(z0)), gc_cal vs the oracle ceiling and the
#    shuffled-label null. If gc_cal ~ ceiling, the correction is LABEL-
#    CALIBRATABLE (a real mechanism).
# P3 FIXED D1 FLOORS: per-pair oracle flips restricted to the frozen-error set.
# P4 TTA as acquisition: corr(TTA instability, |delta_z*|) and corr(TTA, error).
#
# Decisive:
#   P1 gc_rich > 3-feature +0.10  -> rich features carry the correction
#   P2 gc_cal ~ gc_rich           -> LABEL-CALIBRATABLE (real method)
#   P2 gc_cal ~ gc_shuf           -> the gain is noise
#   P4 corr > 0                   -> TTA is an acquisition signal
#
# Usage:
#   DRY_RUN=1 bash run_al_feature_correction.sh 3
#   SMOKE=1   bash run_al_feature_correction.sh 3
#   bash run_al_feature_correction.sh 3
#   CONDS="fog,crosstalk" bash run_al_feature_correction.sh 3
#
# Output:
#   robust_diagnostic/logs/al_feature_correction_dglsspp.json
#   logs/al_feature_correction_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Feature-conditioned correction (8B, DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --tta_augs 3 --knn 5 --knn_sub 2000 --b_labels 2,4 --n_rff 16"
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
  logf="logs/al_feature_correction_${label}.log"
  outjson="robust_diagnostic/logs/al_feature_correction_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_feature_correction_diag.py \
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
  echo "=== FEATURE-CONDITIONED CORRECTION OK ==="
  echo "  P1 gc_rich > +0.10        -> rich features carry the correction"
  echo "  P2 gc_cal ~ gc_rich       -> LABEL-CALIBRATABLE (real method)"
  echo "  P2 gc_cal ~ gc_shuf       -> the gain is noise"
  echo "  P4 corr(tta, |dz*|) > 0   -> TTA is an acquisition signal"
else
  echo "=== FEATURE-CONDITIONED CORRECTION FAILED ==="
  exit 1
fi
