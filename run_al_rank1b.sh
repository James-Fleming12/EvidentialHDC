#!/usr/bin/env bash
# run_al_rank1b.sh: the rank-1-per-label decomposition, CORRECTED for the two
# confounds in Iteration 3 (al_rank1_diag.py).
#
# Fix 1: SPAN-CAPTURE instead of per-label flattened cosine. Does the span of the
#   b labels capture R (||P_span R||/||R||)? This is the right question -- a
#   single label is never the whole residual, but the span can be.
# Fix 2: NORMALIZED directions (u_i/||u_i||) so eta is a bounded trust radius,
#   not a raw 4.8-magnitude overstep.
#
# Decisive reads:
#   capture(b) > 0.5 -> labels DO span the residual; Iteration-3 was mis-scaled.
#   capture(b) ~ 0   -> few labels cannot span R (the real, method-independent
#                       conclusion).
#   keep_oracle_good positive while aggregate negative -> rejection is the lever.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_rank1b.sh 2
#   SMOKE=1   bash run_al_rank1b.sh 2
#   bash run_al_rank1b.sh 2
#   CONDS="fog,crosstalk" bash run_al_rank1b.sh 2
#
# Output:
#   robust_diagnostic/logs/al_rank1b_{dglsspp,covshift_ep10}.json
#   logs/al_rank1b_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8}"
ETA="${ETA:-0.5}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Rank-1 corrected (span-capture + normalized) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS eta=$ETA"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4 --eta 0.5 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_rank1b_${label}.log"
  outjson="robust_diagnostic/logs/al_rank1b_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_rank1b_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --budgets $BUDGETS --eta $ETA $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== RANK-1 CORRECTED OK ==="
  echo "  capture(b) > 0.5 -> labels DO span R; Iteration-3 was mis-scaled."
  echo "  capture(b) ~ 0   -> few labels cannot span R (real, method-independent)."
  echo "  keep_oracle_good positive while aggregate negative -> rejection is the lever."
else
  echo "=== RANK-1 CORRECTED FAILED ==="
  exit 1
fi
