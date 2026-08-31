#!/usr/bin/env bash
# run_al_uest_bdry.sh: the decision-rule U estimators that the al_uest diagnostic
# could not (pool cov / mean shift / CCA are orthogonal to the oracle residual).
#
# FAMILY 1 (label-free, boundary-conditioned): bdry_pca, bdry_outer,
#   bdry_margin_cov, pair_ab -- decision-weighted geometry (near-boundary points,
#   x along the boundary normal, inverse-margin weighting, confused pair).
# FAMILY 2 (few-label, adaptation tangent space): tangent_b (PCA across
#   provisional tiny ridge updates), ensemble (stack of weak-classifier dW).
#
# Runs BOTH DGLSS++ (primary AL target) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_uest_bdry.sh 2
#   SMOKE=1   bash run_al_uest_bdry.sh 2
#   bash run_al_uest_bdry.sh 2
#   CONDS="fog,crosstalk" R_SWEEP="2,4" BUDGET_SWEEP="8,32" bash run_al_uest_bdry.sh 2
#
# Output:
#   robust_diagnostic/logs/al_uest_bdry_{dglsspp,covshift_ep10}.json
#   logs/al_uest_bdry_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4}"
BUDGET_SWEEP="${BUDGET_SWEEP:-8,32}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Boundary / tangent U-estimation | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r_sweep=$R_SWEEP budget_sweep=$BUDGET_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 8 --bdry_frac 0.3 --n_windows 2"
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
  logf="logs/al_uest_bdry_${label}.log"
  outjson="robust_diagnostic/logs/al_uest_bdry_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_uest_bdry_diag.py \
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
  echo "=== BOUNDARY / TANGENT U-ESTIMATION OK ==="
  echo "  FAMILY 1 (label-free): does decision-boundary geometry contain the residual?"
  echo "    align >> 0.1 (vs oracle) reopens the label-free TTA path."
  echo "  FAMILY 2 (few-label): does PCA across provisional updates recover U?"
  echo "    align high + AL chain near ceiling = couple-of-points AL deployable."
  echo "  Compare against oracle-U reference (gc on each cell)."
else
  echo "=== BOUNDARY / TANGENT U-ESTIMATION FAILED ==="
  exit 1
fi
