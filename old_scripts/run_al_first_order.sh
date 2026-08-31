#!/usr/bin/env bash
# run_al_first_order.sh: the decisive test -- is the missing-mass/covariance
# problem an artifact of the RIDGE estimator?
#
# Replaces the Newton/ridge step with FIRST-ORDER / TRUST-REGION updates that
# never estimate the pool covariance (U^T X^T X U). Labels supply DIRECTION; the
# step size is a controlled trust radius; TTA supplies the scale (trust gate).
#
# Methods: oracle_ridge (ref) | oracle_first (C=s*G) | oracle_norm (C=s*G/||G||)
#          | gradspan (U from label gradients) | full_grad (no U)
#          | boundary_pair (confusion-pair margins) | tta_trust (gated rho).
#
# Runs BOTH DGLSS++ (primary) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_first_order.sh 2
#   SMOKE=1   bash run_al_first_order.sh 2
#   bash run_al_first_order.sh 2
#   CONDS="fog,crosstalk" STEP_SWEEP="0.05,0.2" bash run_al_first_order.sh 2
#
# Output:
#   robust_diagnostic/logs/al_first_order_{dglsspp,covshift_ep10}.json
#   logs/al_first_order_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4}"
BUDGET_SWEEP="${BUDGET_SWEEP:-8,32}"
STEP_SWEEP="${STEP_SWEEP:-0.05,0.2,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "First-order / trust-region updates | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP b=$BUDGET_SWEEP step=$STEP_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 8 --step_sweep 0.2"
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
  logf="logs/al_first_order_${label}.log"
  outjson="robust_diagnostic/logs/al_first_order_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_first_order_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --budget_sweep $BUDGET_SWEEP --step_sweep $STEP_SWEEP \
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
  echo "=== FIRST-ORDER / TRUST-REGION OK ==="
  echo "  oracle_first/norm vs oracle_ridge: was the covariance the artifact?"
  echo "  gradspan vs oracle-U: is U estimation necessary?"
  echo "  tta_trust: does the gate keep corrupted gains and reject healthy?"
  echo "  boundary_pair: do confusion-pair margins capture R?"
else
  echo "=== FIRST-ORDER / TRUST-REGION FAILED ==="
  exit 1
fi
