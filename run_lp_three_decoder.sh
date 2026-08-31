#!/usr/bin/env bash
# run_lp_three_decoder.sh: the four-decoder numbers, FULL-HARNESS protocol
# (paper-realistic, docs/lin_probe_training/).
#   no-HDC     the model's own trained 1x1 conv head
#   prototype  mean binarized code per class, cosine decode (README R1)
#   linear     ridge probe on the binarized codes (README R4)
#   raw-linear the same ridge probe on the RAW 128-d features
# Protocol = README: 200k clean reservoir fit (all frames), FULL streaming eval
# (~300M pts/condition), spectral-exact ridge, default severity heavy.
#
# Usage:
#   DRY_RUN=1 bash run_lp_three_decoder.sh 3
#   SMOKE=1   bash run_lp_three_decoder.sh 3
#   bash run_lp_three_decoder.sh 3
#   CONDS="fog,crosstalk" bash run_lp_three_decoder.sh 3
#   SEVS="light,moderate,heavy" bash run_lp_three_decoder.sh 3   # 3-sev mean
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
SEVS="${SEVS:-heavy}"
MAP19="${MAP19:-0}"
CEILING="${CEILING:-0}"
SM_FRAMES="${SM_FRAMES:-30}"
echo "Four-decoder numbers, full-harness protocol (DGLSS++) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS sevs=$SEVS map19=$MAP19 ceiling=$CEILING"
MAP19_ARGS=""
if [ "$MAP19" = "1" ]; then
  MAP19_ARGS="--map19"
  echo "  [map19] GeoID 19-class map, fixed-19 mIoU, no-HDC dropped"
fi
CEIL_ARGS=""
if [ "$CEILING" = "1" ]; then
  CEIL_ARGS="--ceiling"
  echo "  [ceiling] also fit + eval labeled-ceiling decoders (corrupted-pool fit)"
fi
ARCH="${ARCH:-}"
ARCH_ARGS=""
if [ -n "$ARCH" ]; then
  ARCH_ARGS="--arch $ARCH"
  echo "  [arch] $ARCH"
fi

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
CKPT="${CKPT:-}"
METHOD="${METHOD:-supcon_vib_dglsspp}"
if [ -n "$CKPT" ]; then
  DGLSSPP="ckpt|$METHOD|$CKPT"
  echo "  [CKPT] evaluating against: $CKPT (method=$METHOD)"
fi
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--max_frames $SM_FRAMES --clean_fit_n 5000 --sevs moderate"
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
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" --sevs \"$SEVS\" \
    $ARCH_ARGS $MAP19_ARGS $CEIL_ARGS $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== FOUR-DECODER OK (full-harness protocol) ==="
  echo "  mIoU_no_hdc / mIoU_proto / mIoU_linear / mIoU_raw_linear per condition"
else
  echo "=== FOUR-DECODER FAILED ==="
  exit 1
fi

