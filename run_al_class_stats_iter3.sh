#!/usr/bin/env bash
# run_al_class_stats_iter3.sh: Iteration 3 -- the three-arm DECISIVE test for the
# class-statistics reformulation.
#
# ARM A  label-free pool baseline: hard / soft / TTA pseudo-means (the label-free
#        ceiling of the construction). + whitened_mean_err + residual-relevant err.
# ARM B  pool basis + scalar gamma_c in R^K (labels estimate "how much class c
#        should move along the pool-derived direction v_c"), K in {3,5,8}.
# ARM C  pseudo-label confusion correction (labels estimate the C x C matrix Q).
# CORRUP  D_rho: corrupt the oracle mean direction, measure gc(rho) -- how precise
#        must the mean estimate be?
#
# Read:
#   Arm A > B,C -> the label-free route dominates (labels not needed)
#   B or C > A  -> labels ARE useful for a low-dimensional correction
#   corruption  -> gc survives large rho = estimator problem solvable
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter3.sh 3
#   SMOKE=1   bash run_al_class_stats_iter3.sh 3
#   bash run_al_class_stats_iter3.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter3.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter3_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter3_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iteration 3 (three-arm) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_per_class 2,4 --k_sweep 3,5 --tta_augs 3"
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
  logf="logs/al_class_stats_iter3_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter3_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter3_diag.py \
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
  echo "=== CLASS-STATS ITERATION 3 OK ==="
  echo "  Arm A > B,C -> the label-free route dominates (labels not needed)"
  echo "  B or C > A  -> labels ARE useful for a low-dimensional correction"
  echo "  corruption  -> gc survives large rho = estimator problem solvable"
else
  echo "=== CLASS-STATS ITERATION 3 FAILED ==="
  exit 1
fi
