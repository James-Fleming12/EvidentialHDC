#!/usr/bin/env bash
# run_probe_bn_reanchor.sh: how much does a BatchNorm running-stat re-anchor
# recover on the collapsed conditions (fog/crosstalk), relative to the labeled
# ceiling?
#
# Decisive comparison per condition (dgl_kitti):
#   frozen      : W0 (clean) with frozen BN    = baseline zero-shot
#   bn_recal    : W0 with re-estimated BN      = statistic substitution (label-free)
#   ceiling     : W* (pool) with frozen BN     = labeled upper bound
#   bn_recal_W* : W* with re-estimated BN
# Scope: --bn_scope bottleneck (conv_1/conv_2) | late (+layer3/4) | all
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (500 frames, fog+crosstalk)
#
# Usage:
#   DRY_RUN=1 bash run_probe_bn_reanchor.sh 2
#   SMOKE=1   bash run_probe_bn_reanchor.sh 2
#   bash run_probe_bn_reanchor.sh 2
#   BN_SCOPE=bottleneck bash run_probe_bn_reanchor.sh 2
#
# Output: robust_diagnostic/logs/probe_bn_reanchor.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-500}"
CONDS="${CONDS:-fog,crosstalk}"
BN_SCOPE="${BN_SCOPE:-late}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp}"
echo "BN re-anchor probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS, scope=$BN_SCOPE)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_bn_reanchor_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --bn_scope $BN_SCOPE --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_bn_reanchor.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_bn_reanchor.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== BN RE-ANCHOR OK ==="
  echo "  bn_recal_frac_of_gap: fraction of the frozen->ceiling gap the label-free"
  echo "  BN re-anchor closes. ~1.0 = BN recalibration alone reaches the labeled bound."
else
  echo "=== BN RE-ANCHOR FAILED (exit $RC) ==="
  exit $RC
fi
