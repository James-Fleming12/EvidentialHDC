#!/usr/bin/env bash
# run_al_ufree.sh: Iteration 1-UFree -- directly optimize dW with the unlabeled
# pool geometry as a PRIOR, never estimating U.
#
# The residual-subspace closure says U is not recoverable from few labels. This
# tests the proposal: solve min_dW L_L(W0+dW) + lam*R(dW; X_pool) directly, where
# R uses the unlabeled pool geometry. Variants:
#   frozen / oracle_U (bound) / a_grad (known-fail baseline) /
#   tikhonov (plain ridge on labels) / pool_span (dW in pool-eigen span) /
#   pool_penalty (penalize high-variance) / hybrid_first
# Acquisition = margin_tta_div (Iteration-1 winner). b in {2,4,8}.
#
# If a U-free variant approaches oracle_U and beats a_grad, the pool geometry IS
# the missing ingredient. If all stay at a_grad level, U-free is closed.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_ufree.sh 2
#   SMOKE=1   bash run_al_ufree.sh 2
#   bash run_al_ufree.sh 2
#   CONDS="fog,crosstalk" bash run_al_ufree.sh 2
#
# Output:
#   robust_diagnostic/logs/al_ufree_{dglsspp,covshift_ep10}.json
#   logs/al_ufree_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8}"
RHO="${RHO:-0.2}"
REG_LAMBDA="${REG_LAMBDA:-1e-3}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "U-free residual (pool-prior regularizer) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS rho=$RHO reg_lambda=$REG_LAMBDA"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4 --rho 0.2 --reg_lambda 1e-3 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_ufree_${label}.log"
  outjson="robust_diagnostic/logs/al_ufree_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_ufree_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --budgets $BUDGETS --rho $RHO --reg_lambda $REG_LAMBDA \
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
  echo "=== U-FREE RESIDUAL OK ==="
  echo "  Does a pool-geometry regularizer make the label-driven dW work where a_grad"
  echo "  (no prior) failed? Compare tikhonov / pool_span / pool_penalty vs oracle_U."
  echo "  If a U-free variant approaches oracle_U and beats a_grad, the pool geometry"
  echo "  IS the missing ingredient. If all stay at a_grad level, U-free is closed."
else
  echo "=== U-FREE RESIDUAL FAILED ==="
  exit 1
fi
