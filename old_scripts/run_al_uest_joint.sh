#!/usr/bin/env bash
# run_al_uest_joint.sh: joint U,C refinement on top of the tangent-U init.
#
# The boundary/tangent diagnostic found tangent_b8 is the ONLY U estimator that
# recovers the oracle residual (align 0.3-0.5), but its AL chain (leverage-in-U
# then C-solve in a FIXED U) is ~0. This tests the doc's synthesis: instead of
# the separate "discover U, then fit C" pipeline, ALTERNATE C-solve / U-gradient
# / orthogonalize on the labeled points, so the same labels that discovered U
# also push it toward the oracle while fitting C.
#
# Reads: baseline (C-solve in fixed tangent-U) vs joint (T in {1,5,20} iters) vs
# oracle-ref (C-solve in oracle U at the same budget = the ceiling).
# Does refinement (i) move U toward oracle, (ii) raise gc toward the oracle-ref?
#
# Runs BOTH DGLSS++ (primary AL target) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_uest_joint.sh 2
#   SMOKE=1   bash run_al_uest_joint.sh 2
#   bash run_al_uest_joint.sh 2
#   CONDS="fog,crosstalk" LR=1e-2 JOINT_ITERS="5,20" bash run_al_uest_joint.sh 2
#
# Output:
#   robust_diagnostic/logs/al_uest_joint_{dglsspp,covshift_ep10}.json
#   logs/al_uest_joint_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4}"
BUDGET_SWEEP="${BUDGET_SWEEP:-8,32}"
LR="${LR:-1e-2}"
JOINT_ITERS="${JOINT_ITERS:-1,5,20}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Joint U,C refinement | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP b=$BUDGET_SWEEP lr=$LR iters=$JOINT_ITERS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 8 --n_windows 2 --joint_iters 1,5"
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
  logf="logs/al_uest_joint_${label}.log"
  outjson="robust_diagnostic/logs/al_uest_joint_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_uest_joint_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --budget_sweep $BUDGET_SWEEP --lr $LR --joint_iters $JOINT_ITERS \
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
  echo "=== JOINT U,C REFINEMENT OK ==="
  echo "  Does alternating C-solve / U-gradient / QR (from the tangent-U init) turn"
  echo "  the ~0 tangent AL chain into a real gain toward the oracle-ref ceiling?"
  echo "  align_U_final rising = refinement pulls U toward oracle; gc rising = deployable."
else
  echo "=== JOINT U,C REFINEMENT FAILED ==="
  exit 1
fi
