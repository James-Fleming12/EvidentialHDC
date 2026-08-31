#!/usr/bin/env bash
# run_al_oracle_curve.sh: the ORACLE ACQUISITION CURVE -- does the label budget
# itself, or the acquisition algorithm, bound the few-label gain?
#
# Fixed downstream (oracle-U first-order, the only few-label mechanism that
# works); only POINT selection varies: random / margin_tta_div (Iteration-1
# winner) / oracle_error (query the frozen probe's own errors) / oracle_pair
# (query the val-truth top error pairs) / margin_perm (margin_tta_div points but
# PERMUTED labels -- the supervised-content control). Budgets 2,4,8,16,32,64.
#
# Decisive reads:
#   oracle_error >> margin_tta_div  -> a real acquisition gap (find the probe's
#                                      errors) -> Story A
#   oracle_error ~ margin_tta_div   -> gain not concentrated on probe errors
#   margin_perm ~ random            -> the labels' supervised content is real
#   all flat until 32-64            -> Story C: label-starved, not
#                                      algorithm-starved
#   gain_concentration ~ 0.02       -> global W-adaptation is the wrong
#                                      abstraction
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_oracle_curve.sh 3
#   SMOKE=1   bash run_al_oracle_curve.sh 3
#   bash run_al_oracle_curve.sh 3
#   CONDS="fog,crosstalk" bash run_al_oracle_curve.sh 3
#
# Output:
#   robust_diagnostic/logs/al_oracle_curve_{dglsspp,covshift_ep10}.json
#   logs/al_oracle_curve_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8,16,32,64}"
RHO="${RHO:-0.05,0.2,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Oracle acquisition curve | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS rho=$RHO"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4,8 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_oracle_curve_${label}.log"
  outjson="robust_diagnostic/logs/al_oracle_curve_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_oracle_curve_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --budgets $BUDGETS --rho_sweep $RHO $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== ORACLE ACQUISITION CURVE OK ==="
  echo "  oracle_error >> margin_tta_div -> a real acquisition gap (Story A)"
  echo "  oracle_error ~ margin_tta_div  -> gain not concentrated on probe errors"
  echo "  margin_perm ~ random           -> the labels' supervised content is real"
  echo "  all flat until 32-64           -> Story C: label-starved"
else
  echo "=== ORACLE ACQUISITION CURVE FAILED ==="
  exit 1
fi
