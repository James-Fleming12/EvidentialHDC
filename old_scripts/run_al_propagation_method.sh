#!/usr/bin/env bash
# run_al_propagation_method.sh: the METHOD decomposition -- which part of the
# propagated-mean decoder fails, and what to add (DGLSS++ only, fog/crosstalk).
#
# For each anchor SELECTION rule (random / mass B2 / loose A4 / mass+loose):
#   the 2x2 accounting (means x counts):
#     gc_prop       M_prop x C_prop    (the method as-is)
#     gc_oracle_M   M_star x C_prop    (isolates the MEAN error)
#     gc_oracle_C   M_prop x C_star    (isolates the COUNT error)
#     gc_both       M_star x C_star    (pool oracle)
#   plus the fix arms:
#     F1  count-reference correction (toward anchor proportions)
#     F1b count-reference correction (toward a smooth prior)
#     F2  per-class mean fix (worst whitened-error classes -> pseudo-mean)
#     F3  F1+F2 combined (the candidate method)
#   plus per-class precision / count error / worst whitened-error classes.
#
# Decisive:
#   mean_error_cost = gc_oracle_M - gc_prop (dominant -> add a mean fix F2)
#   count_error_cost = gc_oracle_C - gc_prop (dominant -> add a count fix F1)
#   count_ref_cost = gc_both - gc_oracle_M  (C_prop vs C_star reference cost)
#   F3 > gc_prop -> the combined candidate method works.
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_method.sh 3
#   SMOKE=1   bash run_al_propagation_method.sh 3
#   bash run_al_propagation_method.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_method.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_method_dglsspp.json
#   logs/al_propagation_method_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Propagation method decomposition (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 2,4 --loose_mult 2.0 --count_alpha 1.0"
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
  logf="logs/al_propagation_method_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_method_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_method_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
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
  echo "=== PROPAGATION METHOD DECOMPOSITION OK ==="
  echo "  mean_error_cost dominates -> add a mean fix (F2)"
  echo "  count_error_cost dominates -> add a count fix (F1)"
  echo "  F3 > gc_prop -> the combined candidate method works"
else
  echo "=== PROPAGATION METHOD DECOMPOSITION FAILED ==="
  exit 1
fi
