#!/usr/bin/env bash
# run_al_local_update.sh: Iteration 2 (Experiment B) -- can a local/conservative
# correction be driven by the SAME few acquisition-selected labels, WITHOUT oracle U?
#
# Update forms: global_oracle (Iteration-1 ceiling ref, uses oracle U) |
# class_bias (per-class logit bias) | prototype (updated class means) |
# class_pair (only the true,true-pred boundaries move) | local_topK (top-K pairs).
# Acquisition rules: random, margin_tta_div, egl (the Iteration-1 winners).
# Includes snow/wet_ground to check the zero-degradation property (P3).
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_local_update.sh 2
#   SMOKE=1   bash run_al_local_update.sh 2
#   bash run_al_local_update.sh 2
#   CONDS="fog,crosstalk" bash run_al_local_update.sh 2
#
# Output:
#   robust_diagnostic/logs/al_local_update_{dglsspp,covshift_ep10}.json
#   logs/al_local_update_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
BUDGETS="${BUDGETS:-2,4,8}"
RULES="${RULES:-random,margin_tta_div,egl}"
ETA="${ETA:-0.05}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Local update forms (Experiment B) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS budgets=$BUDGETS eta=$ETA rules=$RULES"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --budgets 2,4 --eta 0.05 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_local_update_${label}.log"
  outjson="robust_diagnostic/logs/al_local_update_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_local_update_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --budgets $BUDGETS --eta $ETA --rules \"$RULES\" \
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
  echo "=== LOCAL UPDATE (EXPERIMENT B) OK ==="
  echo "  Can class_bias / prototype / class_pair / local_topK, driven by the same"
  echo "  few labels, reach meaningful gc WITHOUT oracle U? (global_oracle = ceiling ref)"
  echo "  snow/wet_ground check the zero-degradation property (P3)."
else
  echo "=== LOCAL UPDATE FAILED ==="
  exit 1
fi
