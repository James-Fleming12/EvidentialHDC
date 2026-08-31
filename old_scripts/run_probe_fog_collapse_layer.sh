#!/usr/bin/env bash
# run_probe_fog_collapse_layer.sh: WHERE within DGLSS++ does the KITTI-C
# fog/crosstalk collapse originate?
#
# Registers per-stage forward hooks (conv1..conv_2) on DGLSS++, streams clean vs
# KITTI-C fog/crosstalk, and reports per-stage activation mean/variance, satur
# ated/dead unit fractions, and BatchNorm running-stat mismatch. Decides:
#   * first block saturates/zeros  -> fix is at the INPUT
#   * collapse builds gradually    -> a late re-normalization / gate can rescue it
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only, fail on error
#   (default)  full run (200 frames, fog+crosstalk)
#
# Usage:
#   DRY_RUN=1 bash run_probe_fog_collapse_layer.sh 2
#   SMOKE=1   bash run_probe_fog_collapse_layer.sh 2
#   bash run_probe_fog_collapse_layer.sh 2
#
# Output: robust_diagnostic/logs/probe_fog_collapse_layer.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp}"
echo "Per-layer fog-collapse probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_fog_collapse_layer_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_fog_collapse_layer.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_fog_collapse_layer.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== FOG COLLAPSE LAYER OK ==="
  echo "  Look at the conv1/layer1 rows: early sat/dead or BN mismatch => input-side fix."
else
  echo "=== FOG COLLAPSE LAYER FAILED (exit $RC) ==="
  exit $RC
fi
