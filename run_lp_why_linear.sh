#!/usr/bin/env bash
# run_lp_why_linear.sh: WHY the HDC linear classifier beats the prototype on
# every condition (docs/lin_probe_training/). Feature-space + corruption
# diagnostics on the same frozen DGLSS++ features:
#   P5 clean gap (static space property vs corruption collapse)
#   P6 disagreement (where they differ, is the linear probe right?)
#   P1-P4 isotropy / code diversity / centroid separation / mean shift+dispersion
#
# Usage:
#   DRY_RUN=1 bash run_lp_why_linear.sh 3
#   SMOKE=1   bash run_lp_why_linear.sh 3
#   bash run_lp_why_linear.sh 3
#   CONDS="fog,crosstalk" bash run_lp_why_linear.sh 3
#
# Output:
#   robust_diagnostic/logs/lp_why_linear_dglsspp.json
#   logs/lp_why_linear_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
SM_FRAMES="${SM_FRAMES:-5}"
echo "Why-linear diagnostics (DGLSS++) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --fit_clean 3000 --val_size 6000 --geo_sub 3000 --sevs moderate"
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
  logf="logs/lp_why_linear_${label}.log"
  outjson="robust_diagnostic/logs/lp_why_linear_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/lp_why_linear_diag.py \
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
  echo "=== WHY-LINEAR OK ==="
  echo "  P5 clean gap / P6 disagreement / P1-P4 feature-space per condition"
else
  echo "=== WHY-LINEAR FAILED ==="
  exit 1
fi
