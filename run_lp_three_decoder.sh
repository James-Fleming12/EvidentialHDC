#!/usr/bin/env bash
# run_lp_three_decoder.sh: the three-decoder numbers (docs/lin_probe_training/).
#   no-HDC   the model's own trained 1x1 conv head
#   prototype  mean binarized code per class, cosine decode
#   linear     ridge probe on the binarized codes (the reference decoder)
# Both HDC decoders fit on clean only (zero-shot); evaluated on clean + each
# condition at light/moderate/heavy with the 3-sev mean per condition.
#
# Usage:
#   DRY_RUN=1 bash run_lp_three_decoder.sh 3
#   SMOKE=1   bash run_lp_three_decoder.sh 3
#   bash run_lp_three_decoder.sh 3
#   CONDS="fog,crosstalk" bash run_lp_three_decoder.sh 3
#
# Output:
#   robust_diagnostic/logs/lp_three_decoder_dglsspp.json
#   logs/lp_three_decoder_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
SM_FRAMES="${SM_FRAMES:-5}"
echo "Three-decoder numbers (DGLSS++) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --fit_clean 3000 --val_size 6000 --sevs moderate"
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
  logf="logs/lp_three_decoder_${label}.log"
  outjson="robust_diagnostic/logs/lp_three_decoder_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/lp_three_decoder_diag.py \
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
  echo "=== THREE-DECODER OK ==="
  echo "  mIoU_no_hdc vs mIoU_proto vs mIoU_linear per condition (3-sev mean)"
else
  echo "=== THREE-DECODER FAILED ==="
  exit 1
fi
