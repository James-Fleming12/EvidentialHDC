#!/usr/bin/env bash
# run_al_uest.sh: estimate the residual subspace U with and without oracle
# labels, and evaluate the full update chain for BOTH goals:
#   GOAL A (TTA, label-free U): a label-free U + pseudo-label C in the
#       U-subspace -- does it finally give a meaningful improvement on
#       fog/crosstalk? (the low-rank constraint rescuing label-free TTA)
#   GOAL B (AL, few-label U): a few-label U + leverage-selected true labels --
#       does it approach the oracle ceiling (the couple-of-points regime)?
#
# U estimators: oracle | softshift, poolcov, ccameans (label-free)
#               | subfit_b, shiftsub_b (b labels, frozen-influence selection)
#
# Runs BOTH plain DGLSS++ (primary AL target: big closeable gap) and cov-shift
# (current-method comparison).
#
# Usage:
#   DRY_RUN=1 bash run_al_uest.sh 2
#   SMOKE=1   bash run_al_uest.sh 2
#   bash run_al_uest.sh 2
#   CONDS="fog,crosstalk" R_SWEEP="2,4" BUDGET_SWEEP="8,32" bash run_al_uest.sh 2
#   EXTRACTORS_OVERRIDE="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp" \
#                       bash run_al_uest.sh 2     # dglsspp only
#
# Output:
#   robust_diagnostic/logs/al_uest_{dglsspp,covshift_ep10}.json
#   logs/al_uest_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4,8}"
BUDGET_SWEEP="${BUDGET_SWEEP:-8,32}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "U-estimation (TTA + AL) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r_sweep=$R_SWEEP budget_sweep=$BUDGET_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 8 --cca_pca_k 4"
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
  logf="logs/al_uest_${label}.log"
  outjson="robust_diagnostic/logs/al_uest_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_uest_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --budget_sweep $BUDGET_SWEEP $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== U-ESTIMATION OK ==="
  echo "  align: which U estimator recovers the oracle residual subspace?"
  echo "  GOAL A: does label-free U + pseudo-label C beat frozen on fog/crosstalk?"
  echo "  GOAL B: does few-label U + leverage true labels approach the ceiling?"
else
  echo "=== U-ESTIMATION FAILED ==="
  exit 1
fi
