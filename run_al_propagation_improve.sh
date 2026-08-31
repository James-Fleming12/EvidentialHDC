#!/usr/bin/env bash
# run_al_propagation_improve.sh: the improvement-axes diagnostic for the
# PROPAGATED-MEAN decoder (DGLSS++ only, fog/crosstalk).
#
# A. MEAN ESTIMATOR: A1 true-means x prop-counts (the clean count split),
#    A2 128-d mean aggregation, A3 agreement-gated means, A4 per-class budget,
#    A5 soft propagation.
# B. AL SELECTION: B1 confidence, B2 mass-stratified, B3 boundary-avoiding
#    anchors vs random.
# C. UPDATE: C1 fractional whitening, C2 norm-constrained step, C3 mean
#    shrinkage toward the pseudo-mean.
#
# All vs gcP (the current method) and W_mean_oracle (the ceiling).
#
# Decisive:
#   A1 ~ W_mean_oracle -> the COUNTS are the gap, not the means
#   A1 << ceiling       -> the MEANS are the real bottleneck
#   A2/A3/A4/A5, B1-B3, C1-C3 > gcP -> a better estimator/selection/update exists
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_improve.sh 3
#   SMOKE=1   bash run_al_propagation_improve.sh 3
#   bash run_al_propagation_improve.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_improve.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_improve_dglsspp.json
#   logs/al_propagation_improve_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Propagation improvement axes (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 2 --beta_sweep 0.5,1.0 --c_sweep 1.0 --shrink_sweep 0.5 --k_eig 128"
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
  logf="logs/al_propagation_improve_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_improve_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_improve_diag.py \
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
  echo "=== PROPAGATION IMPROVEMENT AXES OK ==="
  echo "  A1 ~ W_mean_oracle -> counts are the gap; << -> means are the bottleneck"
  echo "  A2/A3/A4/A5, B1-B3, C1-C3 > gcP -> a better piece exists (use it)"
else
  echo "=== PROPAGATION IMPROVEMENT AXES FAILED ==="
  exit 1
fi
