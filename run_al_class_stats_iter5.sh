#!/usr/bin/env bash
# run_al_class_stats_iter5.sh: Iteration 5 -- the COEFFICIENT-ESTIMATION
# diagnostic (the single remaining bottleneck of the class-statistics
# reformulation).
#
# A. THE WINDOW: gc(gamma) curve for a global scalar on the top-K classes +
#    per-class oracle gamma* (is scalar estimation THE whole problem?).
# B. THE ESTIMATORS: raw / shrink1 / gamma1 / normscale / oracle, per budget.
#
# Decisive reads:
#   gc(gamma*_perclass) ~ W_mean_oracle -> scalar estimation is THE problem
#   an estimator gc near the curve's peak -> the mechanism works
#   gamma_hat vs gamma* -> the bias (systematically too small/large?)
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter5.sh 3
#   SMOKE=1   bash run_al_class_stats_iter5.sh 3
#   bash run_al_class_stats_iter5.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter5.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter5_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter5_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iteration 5 (coefficient estimation) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_per_class 2,4 --gamma_sweep 0,0.5,1.0,2.0 --alpha_sweep 1"
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
  logf="logs/al_class_stats_iter5_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter5_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter5_diag.py \
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
  echo "=== CLASS-STATS ITERATION 5 OK ==="
  echo "  gc(gamma*_perclass) ~ W_mean_oracle -> scalar estimation is THE problem"
  echo "  an estimator gc near the curve peak -> the mechanism works"
else
  echo "=== CLASS-STATS ITERATION 5 FAILED ==="
  exit 1
fi
