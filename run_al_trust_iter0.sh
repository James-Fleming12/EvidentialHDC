#!/usr/bin/env bash
# run_al_trust_iter0.sh: Iteration 0 -- validate the three assumptions the
# trust-region AL method depends on (active_iterations_3.md), BEFORE building the
# full pipeline.
#
# A1 coarse-U robustness: does the normalized trust-region step work with the
#    tangent-b8 U (align 0.3-0.5), not just oracle U? tangent ~ oracle >> random
#    -> U deployable; tangent ~ random -> U is the blocker.
# A2 gate validity: is a label-free gauge (conf_drop / mean_shift_cos /
#    r4_r1_disagree) rank-correlated with the useful rho across conditions?
# A3 accept/reject: does a label-free score (d_conf / d_disagree / comb) separate
#    positive-gc from negative-gc updates?
#
# Runs BOTH DGLSS++ (primary) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_trust_iter0.sh 2
#   SMOKE=1   bash run_al_trust_iter0.sh 2
#   bash run_al_trust_iter0.sh 2
#   CONDS="fog,crosstalk" R=2 B=8 bash run_al_trust_iter0.sh 2
#
# Output:
#   robust_diagnostic/logs/al_trust_iter0_{dglsspp,covshift_ep10}.json
#   logs/al_trust_iter0_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R="${R:-2}"
B="${B:-8}"
RHO_SWEEP="${RHO_SWEEP:-0.01,0.05,0.1,0.2,0.4,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Trust-region Iter0 (assumption validation) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R b=$B rho=$RHO_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --n_windows 2 --rho_sweep 0.05,0.2"
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
  logf="logs/al_trust_iter0_${label}.log"
  outjson="robust_diagnostic/logs/al_trust_iter0_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_trust_iter0_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r $R --b $B --rho_sweep $RHO_SWEEP $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== TRUST-REGION ITER0 (ASSUMPTIONS) OK ==="
  echo "  A1: tangent vs oracle vs random U gc-vs-rho (is coarse U enough?)"
  echo "  A2: gauge vs oracle_best_gc spearman across conditions (is the gate valid?)"
  echo "  A3: d_conf / d_disagree / comb vs gc (is accept/reject viable?)"
else
  echo "=== TRUST-REGION ITER0 FAILED ==="
  exit 1
fi
