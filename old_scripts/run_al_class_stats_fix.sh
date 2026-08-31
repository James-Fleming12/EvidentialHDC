#!/usr/bin/env bash
# run_al_class_stats_fix.sh: validate the THREE fix paths from the class-stats
# Iteration 1 (class_stats_iters.md), all white-box, before other methods.
#
# FIX 1. Pool pseudo-label means + few-label bias correction (b x alpha_d).
# FIX 2. Mean-SHIFT-only estimation with shrinkage s (step on the shift).
# FIX 3. Regularized whitening: lambda_whitening sweep + rank truncation, on the
#        same estimated means (isolates the whitening from the estimator).
#
# All compared to the SAME references: W0 (0), W_mean_oracle (ceiling), W* (1).
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_fix.sh 3
#   SMOKE=1   bash run_al_class_stats_fix.sh 3
#   bash run_al_class_stats_fix.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_fix.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_fix_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_fix_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats fix-path validation | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_per_class 2,4 --rank_sweep 64,256"
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
  logf="logs/al_class_stats_fix_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_fix_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_fix_diag.py \
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
  echo "=== CLASS-STATS FIX-PATH VALIDATION OK ==="
  echo "  fix1 ~ W_mean_oracle at small b -> pool pseudo-means + bias correction works"
  echo "  fix2 positive for some s        -> shift-only with shrinkage is the right object"
  echo "  fix3 positive for some lam_w/rank -> the whitening was the amplifier"
else
  echo "=== CLASS-STATS FIX-PATH VALIDATION FAILED ==="
  exit 1
fi
