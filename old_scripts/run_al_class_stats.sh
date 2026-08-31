#!/usr/bin/env bash
# run_al_class_stats.sh: the CLASS-STATISTICS decoder reformulation -- naive,
# WHITE-BOX tests on the W = Sigma^-1 P M decomposition.
#
# A. The decomposition: R_mean_frac (how much of R is the class-mean shift) +
#    decoder ladder (W0 / proto_oracle / W_mean_oracle / W*).
# B. Few-label mean re-estimation: b per class x shrinkage alpha x selection,
#    gc + per-class mean error (the update works? a DIFFERENT object than the
#    flat Iteration-7 curve).
# C. Selection ablation: random / proto_dist / entropy / oracle_error.
# D. Update details: softmax temperature + top-K update scope.
#
# Read:
#   R_mean_frac ~ 1 -> the mean-shift IS the residual; full ceiling for this
#   W_est gc ~ W_mean_oracle at small b -> few labels re-estimate the means well
#   proto << whitened -> whitening is essential (why R1 closed but this need not)
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats.sh 3
#   SMOKE=1   bash run_al_class_stats.sh 3
#   bash run_al_class_stats.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-statistics decoder (white-box) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_per_class 2,4 --temp_sweep 1,2"
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
  logf="logs/al_class_stats_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_diag.py \
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
  echo "=== CLASS-STATISTICS DECODER OK ==="
  echo "  R_mean_frac ~ 1            -> the mean-shift IS the residual (full ceiling)"
  echo "  W_est ~ W_mean_oracle      -> few labels re-estimate the means well"
  echo "  proto << whitened          -> whitening is essential"
else
  echo "=== CLASS-STATISTICS DECODER FAILED ==="
  exit 1
fi
