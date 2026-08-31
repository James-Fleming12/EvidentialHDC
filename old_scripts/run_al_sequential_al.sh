#!/usr/bin/env bash
# run_al_sequential_al.sh: A3 -- SEQUENTIAL AL (labels reveal the next query).
#
# Same downstream (oracle-U first-order, the only few-label mechanism that works)
# for every acquisition arm; only the acquisition varies. One-shot rules:
# random + margin_tta_div (Iteration-1 winner). Sequential rules:
# seq_margin (adaptive ordering, no pair focus), seq_pair (focus the next query
# on a revealed error pair), seq_pair_tta / seq_pair_div (pair focus + TTA /
# diversity).
#
# Decisive reads:
#   seq_* > margin_tta_div  -> sequential labels reveal where to look next
#   seq_* ~ margin_tta_div  -> the adaptive loop adds nothing over one-shot
#   seq_* < random          -> the loop actively hurts
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_sequential_al.sh 3
#   SMOKE=1   bash run_al_sequential_al.sh 3
#   bash run_al_sequential_al.sh 3
#   CONDS="fog,crosstalk" bash run_al_sequential_al.sh 3
#
# Output:
#   robust_diagnostic/logs/al_sequential_al_{dglsspp,covshift_ep10}.json
#   logs/al_sequential_al_{label}.log

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
echo "Sequential AL (A3) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS rho=$RHO"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_sequential_al_${label}.log"
  outjson="robust_diagnostic/logs/al_sequential_al_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_sequential_al_diag.py \
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
  echo "=== SEQUENTIAL AL (A3) OK ==="
  echo "  seq_* > margin_tta_div  -> sequential labels reveal where to look next"
  echo "  seq_* ~ margin_tta_div  -> the adaptive loop adds nothing over one-shot"
  echo "  seq_* < random          -> the loop actively hurts"
else
  echo "=== SEQUENTIAL AL (A3) FAILED ==="
  exit 1
fi
