#!/usr/bin/env bash
# run_al_pool_basis.sh: P1 -- POOL-DERIVED BASIS + FEW-LABEL COEFFICIENT
# SELECTION (the last open parameter-update branch, new_iters.md Tier 1.5).
#
# Iteration 3b closed "few labels -> span(x_i) -> Delta W" (the label span
# captures 0.4-2.0% of R). Iteration 1 showed the COEFFICIENT half is easy given
# oracle U (+0.29-0.37 gc). This test asks the one open question: can the
# UNLABELED POOL provide a basis that CONTAINS R (basis half, via span-capture +
# oracle-coefficient ceiling), and can a few labels SELECT/WEIGHT the right
# combination from it (selection half: firstorder / lsq vs oracle coefficients)?
#
# Decisive reads:
#   span_capture(full) ~ 0            -> no pool structure contains R; P1 closed
#   oracle_coef gc ~ oracle_U ref gc  -> the dictionary IS as good as R's top-2
#   selection gc ~ oracle_coef gc     -> few labels pick the right combination
#   selection gc << oracle_coef gc    -> basis ok but labels can't select it
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_pool_basis.sh 3
#   SMOKE=1   bash run_al_pool_basis.sh 3
#   bash run_al_pool_basis.sh 3
#   CONDS="fog,crosstalk" bash run_al_pool_basis.sh 3
#
# Output:
#   robust_diagnostic/logs/al_pool_basis_{dglsspp,covshift_ep10}.json
#   logs/al_pool_basis_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8}"
RHO="${RHO:-0.05,0.2,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Pool-basis + label-selection (P1) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS rho=$RHO"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4 --cand_frac 0.2 --tta_augs 3 --r 2 --n_pairs 4"
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
  logf="logs/al_pool_basis_${label}.log"
  outjson="robust_diagnostic/logs/al_pool_basis_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_pool_basis_diag.py \
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
  echo "=== POOL-BASIS (P1) OK ==="
  echo "  span_capture(full) ~ 0        -> no pool structure contains R; P1 closed at basis"
  echo "  oracle_coef ~ oracle_U ref gc -> the pool dictionary IS as good as R's top-2"
  echo "  selection ~ oracle_coef gc    -> few labels pick the right combination (P1 wins)"
else
  echo "=== POOL-BASIS (P1) FAILED ==="
  exit 1
fi
