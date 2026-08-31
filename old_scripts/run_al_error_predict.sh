#!/usr/bin/env bash
# run_al_error_predict.sh: the ERROR-PREDICTABILITY diagnostic -- does ANY cheap
# label-free statistic predict the frozen probe's oracle errors? The outcome
# SELECTS the next update mechanism (boundary-gate vs class-mean decoder vs
# mixture-of-decoders vs propagation vs reformat-what-is-stored) -- not a stop
# signal.
#
# Features (all label-free): margin, entropy, p1, p2, tta_var, tta_ent,
# tta_agree, proto_dist, proto_disagree, density, local_disagree,
# classifier_div + a 4-feature logistic combination ceiling.
#
# Read: enrichment@top1% >> 1 on a feature -> that statistic finds the errors
# and should drive the update mechanism.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_error_predict.sh 3
#   SMOKE=1   bash run_al_error_predict.sh 3
#   bash run_al_error_predict.sh 3
#   CONDS="fog,crosstalk" bash run_al_error_predict.sh 3
#
# Output:
#   robust_diagnostic/logs/al_error_predict_{dglsspp,covshift_ep10}.json
#   logs/al_error_predict_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Error-predictability diagnostic | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --eval_size 4000 --max_clean 5000 --tta_augs 3 --knn 5"
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
  logf="logs/al_error_predict_${label}.log"
  outjson="robust_diagnostic/logs/al_error_predict_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_error_predict_diag.py \
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
  echo "=== ERROR-PREDICTABILITY OK ==="
  echo "  enrichment@top1% >> 1 -> that statistic finds the errors; use it to"
  echo "                          drive the next update mechanism"
  echo "  ALL flat -> the update must REFORMAT what is stored/updated"
else
  echo "=== ERROR-PREDICTABILITY FAILED ==="
  exit 1
fi
