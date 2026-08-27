#!/usr/bin/env bash
# run_al_min_label_residual.sh: how few TRUE labels can the low-rank residual
# update W_res = W0 + U_r C use? Sweeps the label budget b in {2,4,8,16,32,56}
# across the residual rank r in {2,4,8}, with random / leverage-in-U /
# per-class selection, per condition. Oracle U_r (to isolate the LABEL-count
# bottleneck from the U-estimation bottleneck).
#
# This is the decisive experiment for "meaningful updates with a couple of
# points": at what (r, b) does the update first EXCEED the frozen decoder, and
# is the bottleneck the label count or the rank?
#
# Usage:
#   DRY_RUN=1 bash run_al_min_label_residual.sh 2
#   SMOKE=1   bash run_al_min_label_residual.sh 2
#   bash run_al_min_label_residual.sh 2
#   CONDS="fog,crosstalk" R_SWEEP="2,4" BUDGET_SWEEP="2,4,8,16" bash run_al_min_label_residual.sh 2
#
# Output:
#   robust_diagnostic/logs/al_min_label_residual_covshift_ep10.json
#   logs/al_min_label_residual_covshift_ep10.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4,8}"
BUDGET_SWEEP="${BUDGET_SWEEP:-2,4,8,16,32,56}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Min-label residual update | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r_sweep=$R_SWEEP budget_sweep=$BUDGET_SWEEP"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
LABEL="covshift_ep10"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 2,4,8"
  echo "  [SMOKE] $SMOKE_ARGS"
fi

logf="logs/al_min_label_residual_${LABEL}.log"
outjson="robust_diagnostic/logs/al_min_label_residual_${LABEL}.json"
CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_min_label_residual_diag.py \
  --path_b \"$CKPT\" --method_b \"$METHOD\" --label \"$LABEL\" --conds \"$CONDS\" \
  --r_sweep $R_SWEEP --budget_sweep $BUDGET_SWEEP $SMOKE_ARGS --out \"$outjson\""
echo "  CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
  exit 0
fi

if eval "$CMD" 2>&1 | tee "$logf"; then
  echo "=== MIN-LABEL RESIDUAL OK ==="
  echo "  -> $outjson"
else
  echo "=== MIN-LABEL RESIDUAL FAILED (exit $?) -- tail of $logf:"
  tail -25 "$logf"
  exit 1
fi
