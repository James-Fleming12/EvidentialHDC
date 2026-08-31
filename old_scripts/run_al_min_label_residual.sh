#!/usr/bin/env bash
# run_al_min_label_residual.sh: how few TRUE labels can the low-rank residual
# update W_res = W0 + U_r C use? Sweeps the label budget b in {2,4,8,16,32,56}
# across the residual rank r in {2,4,8}, with random / leverage-in-U /
# per-class selection, per condition. Oracle U_r (to isolate the LABEL-count
# bottleneck from the U-estimation bottleneck).
#
# EXTRACTORS: by default runs BOTH plain DGLSS++ (supcon_vib_dglsspp) and
# cov-shift (supcon_vib_dglsspp_inputin_in_chan). DGLSS++ is the primary AL
# target: on KITTI-C 3-sev it has the big closeable gap (fog zs 22.5 -> ceil
# 35.2, +12.7; crosstalk +17.5) where AL actually has headroom to recover,
# while cov-shift's frozen is already within +2.9 of its ceiling. cov-shift is
# kept as the current-method comparison. Use EXTRACTORS_OVERRIDE to run one.
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
#   EXTRACTORS_OVERRIDE="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp" \
#                       bash run_al_min_label_residual.sh 2     # dglsspp only
#
# Output:
#   robust_diagnostic/logs/al_min_label_residual_{dglsspp,covshift_ep10}.json
#   logs/al_min_label_residual_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4,8}"
BUDGET_SWEEP="${BUDGET_SWEEP:-2,4,8,16,32,56}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Min-label residual update | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r_sweep=$R_SWEEP budget_sweep=$BUDGET_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --budget_sweep 2,4,8"
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
  logf="logs/al_min_label_residual_${label}.log"
  outjson="robust_diagnostic/logs/al_min_label_residual_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_min_label_residual_diag.py \
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
  echo "=== MIN-LABEL RESIDUAL OK ==="
  echo "  dglsspp is the PRIMARY AL target (big closeable gap); covshift_ep10 is the"
  echo "  current-method comparison. Compare at which (r, b) each first exceeds frozen,"
  echo "  and whether leverage_u beats random/per_class."
else
  echo "=== MIN-LABEL RESIDUAL FAILED ==="
  exit 1
fi
