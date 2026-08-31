#!/usr/bin/env bash
# run_al_error_loop.sh: A5 -- the ERROR-CORRECTION AL LOOP (the recommended next
# test, new_iters.md Level 2 / local decision correction).
#
# Labels reveal recurring (pred,true) error pairs; subsequent queries focus on
# those pairs' boundaries (the sequential loop); the "repair" is a DECODE-TIME
# re-ranking/gating of the identified pairs -- NOT a W update.
#
# Arms:
#   pair_bias  per-pair logit offset (B3 generalized to pairs) -- label pairs vs
#              oracle pairs vs random pairs (control)
#   pair_gate  margin-threshold flip -- label pairs vs oracle pairs
# Plus pair-discovery precision/recall vs the val-truth error pairs.
#
# Decisive reads:
#   label ~ oracle >> 0 -> the error-correction loop works end to end.
#   label << oracle     -> pair discovery is the bottleneck.
#   random ~ label      -> the correction is just noise.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_error_loop.sh 3
#   SMOKE=1   bash run_al_error_loop.sh 3
#   bash run_al_error_loop.sh 3
#   CONDS="fog,crosstalk" bash run_al_error_loop.sh 3
#
# Output:
#   robust_diagnostic/logs/al_error_loop_{dglsspp,covshift_ep10}.json
#   logs/al_error_loop_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Error-correction AL loop (A5) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS"

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
  logf="logs/al_error_loop_${label}.log"
  outjson="robust_diagnostic/logs/al_error_loop_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_error_loop_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --budgets $BUDGETS $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== ERROR-CORRECTION AL LOOP (A5) OK ==="
  echo "  label ~ oracle >> 0 -> the loop works end to end"
  echo "  label << oracle     -> pair discovery is the bottleneck"
  echo "  random ~ label      -> the correction is just noise"
else
  echo "=== ERROR-CORRECTION AL LOOP (A5) FAILED ==="
  exit 1
fi
