#!/usr/bin/env bash
# run_al_propagation_validate.sh: validation of the propagated-mean result
# (Iteration 10) -- DGLSS++ ONLY, the extractor with a real ceiling.
#
# V1 true oracle-count ceiling (fix the Iteration-10 gcO bug: CLEAN counts ->
#    now ORACLE counts C_star): how much of the +0.73/+0.99 ceiling the
#    propagated MEANS capture vs what is lost to count error.
# V2 clean-source bank mean decoder (label-free memory-bank idea).
# V3 influence vs random anchor selection (per-budget, repeated for variance).
# V4 per-class propagated precision breakdown.
# V5 count error per budget vs oracle-count gc (means vs counts split).
#
# Decisive:
#   V1 oracle-count gc ~ W_mean_oracle -> the MEANS capture the ceiling; counts
#      are the only gap. << ceiling -> means are still the bottleneck.
#   V2 ~ V1 (label-free) -> the method needs NO labels at all.
#   V3 influence > random -> use influence selection going forward.
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_validate.sh 3
#   SMOKE=1   bash run_al_propagation_validate.sh 3
#   bash run_al_propagation_validate.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_validate.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_validate_dglsspp.json
#   logs/al_propagation_validate_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Propagation validation (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 2,4 --clean_bank 5000 --reps 2"
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
  logf="logs/al_propagation_validate_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_validate_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_validate_diag.py \
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
  echo "=== PROPAGATION VALIDATION OK ==="
  echo "  V1 oracle-count gc ~ W_mean_oracle -> the MEANS capture the ceiling;"
  echo "     counts are the only gap (means-vs-counts split now correct)"
  echo "  V2 clean-source (label-free) ~ V1 -> the method needs NO labels"
  echo "  V3 influence > random -> use influence selection"
else
  echo "=== PROPAGATION VALIDATION FAILED ==="
  exit 1
fi
