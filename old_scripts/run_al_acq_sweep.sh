#!/usr/bin/env bash
# run_al_acq_sweep.sh: Iteration 1 (Experiment A) -- does better label selection
# compensate for the lack of U?
#
# The residual-subspace closure says U (the oracle update direction) is not
# obtainable from 2-8 labels. This isolates the remaining question: with the
# DOWNSTREAM UPDATE FIXED (normalized first-order + oracle U), does the ACQUISITION
# RULE matter? Compare random / margin / entropy / tta_inst / margin_tta /
# margin_div / tta_div / margin_tta_div / class_pair / egl at b in {2,4,8}.
#
# If a rule beats random at b=4-8 by a real margin, active selection compensates
# for the label budget without U. If all tie, the bottleneck is the update and
# Experiment B (local corrections) is next.
#
# Runs BOTH DGLSS++ (primary) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_acq_sweep.sh 2
#   SMOKE=1   bash run_al_acq_sweep.sh 2
#   bash run_al_acq_sweep.sh 2
#   RULES="random,margin,class_pair" BUDGETS="2,4,8" bash run_al_acq_sweep.sh 2
#
# Output:
#   robust_diagnostic/logs/al_acq_sweep_{dglsspp,covshift_ep10}.json
#   logs/al_acq_sweep_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
R="${R:-2}"
BUDGETS="${BUDGETS:-2,4,8}"
RHO_SWEEP="${RHO_SWEEP:-0.05,0.2,0.8}"
RULES="${RULES:-random,margin,entropy,tta_inst,margin_tta,margin_div,tta_div,margin_tta_div,class_pair,egl}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Acquisition-function sweep | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R budgets=$BUDGETS rules=$RULES"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r 2 --budgets 2,4 --rho_sweep 0.2 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_acq_sweep_${label}.log"
  outjson="robust_diagnostic/logs/al_acq_sweep_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_acq_sweep_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r $R --budgets $BUDGETS --rho_sweep $RHO_SWEEP --rules \"$RULES\" \
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
  echo "=== ACQUISITION SWEEP OK ==="
  echo "  Does margin/TTA/class_pair beat random at b=4-8 by a real margin?"
  echo "  yes -> active selection compensates for the budget without U."
  echo "  all tie -> bottleneck is the update; Experiment B (local corrections) next."
else
  echo "=== ACQUISITION SWEEP FAILED ==="
  exit 1
fi
