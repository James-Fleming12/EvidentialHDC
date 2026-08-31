#!/usr/bin/env bash
# run_al_bank_floor.sh: Iteration 2 -- how much supervision is actually needed to
# reach a WORKING U? The bank-size/selection -> U-quality floor curve.
#
# Sweeps bank size (28..556) and selection rule (random / per_class /
# leverage_oracle / margin_frozen), fits W_sub on each bank, U_bk = SVD(W_sub-W0),
# and measures (a) align to oracle U and (b) the trust-region step's gc.
#
# leverage_oracle is the UPPER BOUND on selection (uses oracle U); if even it
# cannot reach the floor cheaply, no deployable rule can.
#
# The answer determines the bank's design (point count + selection + what to store)
# before building compression/streaming machinery.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_bank_floor.sh 2
#   SMOKE=1   bash run_al_bank_floor.sh 2
#   bash run_al_bank_floor.sh 2
#   CONDS="fog,crosstalk" BANK_SIZES="28,106,556" bash run_al_bank_floor.sh 2
#
# Output:
#   robust_diagnostic/logs/al_bank_floor_{dglsspp,covshift_ep10}.json
#   logs/al_bank_floor_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
R_SWEEP="${R_SWEEP:-2,4}"
BANK_SIZES="${BANK_SIZES:-28,56,106,156,356,556}"
B_DIRECTION="${B_DIRECTION:-8}"
RHO_SWEEP="${RHO_SWEEP:-0.05,0.2,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Bank floor (supervision -> U quality) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP bank_sizes=$BANK_SIZES b_direction=$B_DIRECTION"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --bank_sizes 28,56 --b_direction 8 --rho_sweep 0.2"
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
  logf="logs/al_bank_floor_${label}.log"
  outjson="robust_diagnostic/logs/al_bank_floor_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_bank_floor_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --bank_sizes $BANK_SIZES --b_direction $B_DIRECTION \
    --rho_sweep $RHO_SWEEP $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== BANK FLOOR OK ==="
  echo "  The floor: smallest N where align > ~0.7 AND gc approaches oracle-U gc."
  echo "  leverage_oracle is the selection UPPER BOUND; if it cannot reach the floor"
  echo "  cheaply, no deployable rule can. ~500 -> bank not cheap; ~56-156 -> viable."
else
  echo "=== BANK FLOOR FAILED ==="
  exit 1
fi
