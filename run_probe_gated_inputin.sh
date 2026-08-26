#!/usr/bin/env bash
# run_probe_gated_inputin.sh: test-time adaptation for KITTI-C fog/crosstalk by
# GATING a per-scan input re-anchor (channels {0,4}) on the BN-mismatch detector.
#
# On a PLAIN DGLSS++ extractor, per scan: bn_mismatch_conv_1 > tau -> apply
# per-scan input-IN to channels {0,4}; else raw. The goal: rescue fog/crosstalk
# (input-bound collapse) WITHOUT paying the always-on input-IN clean-capacity
# cost on healthy scans. Compares raw / gated / always_on / labeled ceiling.
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (200 frames, fog+crosstalk+snow)
#
# Usage:
#   DRY_RUN=1 bash run_probe_gated_inputin.sh 2
#   SMOKE=1   bash run_probe_gated_inputin.sh 2
#   bash run_probe_gated_inputin.sh 2
#   TAU=0.5   bash run_probe_gated_inputin.sh 2   # manual threshold
#
# Output: robust_diagnostic/logs/probe_gated_inputin.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk,snow}"
SM_FRAMES="${SM_FRAMES:-10}"
TAU="${TAU:-}"
EXTRACTORS="${EXTRACTORS:-dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp}"
echo "Gated input-IN TTA probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS, tau=$TAU)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

TAU_FLAG=""
if [ -n "$TAU" ]; then TAU_FLAG="--tau $TAU"; fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_gated_inputin_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS $TAU_FLAG --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_gated_inputin.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_gated_inputin.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== GATED INPUT-IN OK ==="
  echo "  fog/crosstalk: gated should recover toward the ceiling (input-bound collapse)."
  echo "  snow: gated should stay at raw (no false-positive re-anchoring of healthy scans)."
else
  echo "=== GATED INPUT-IN FAILED (exit $RC) ==="
  exit $RC
fi
