#!/usr/bin/env bash
# run_al_class_stats_iter4.sh: Iteration 4 -- push the two green-light findings
# from Iteration 3 (robust direction + Arm C confusion correction).
#
# PART 1  refined Arm C (confusion correction): C0 base, C1 pool-regularized Q
#         (shrink toward the pool pseudo prior), C2 iterated self-training,
#         + Q estimation error vs the full-pool oracle Q.
# PART 2  Arm B with alternative pool directions: pseudo (Iteration-3 failure),
#         highconf (confident core), density (mode core). + oracle direction
#         alignment (which direction points at the true shift).
# PART 3  class-prior correction: W = Sigma^-1 M^T P with P from the pool
#         (P_pseudo label-free, P_oracle ceiling).
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter4.sh 3
#   SMOKE=1   bash run_al_class_stats_iter4.sh 3
#   bash run_al_class_stats_iter4.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter4.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter4_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter4_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iteration 4 | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_per_class 2,4 --k_sweep 3,5 --n_iter 1"
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
  logf="logs/al_class_stats_iter4_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter4_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter4_diag.py \
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
  echo "=== CLASS-STATS ITERATION 4 OK ==="
  echo "  C1/C2 > C0           -> the 8-label Q noise was the limiter; pool prior /"
  echo "                           self-training helps"
  echo "  dir_align highconf/density > pseudo -> a better pool direction exists"
  echo "  P_oracle ~ W_mean_oracle            -> the prior term matters"
else
  echo "=== CLASS-STATS ITERATION 4 FAILED ==="
  exit 1
fi
