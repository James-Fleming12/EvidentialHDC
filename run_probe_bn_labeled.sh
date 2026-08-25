#!/usr/bin/env bash
# run_probe_bn_labeled.sh: with LABELS, does updating the BN layer itself (fitting
# the per-channel affine gamma/beta so corrupted per-class activations map to
# clean per-class means) recover the fog/crosstalk collapse?
#
# The label-free BN re-anchor (probe_bn_reanchor) was NEGATIVE. This is the
# labeled oracle: fit the BN affine with per-class labels, decode with W0, and
# compare to the labeled ceiling. Also reports pre-BN class separability (clean
# vs corrupted) -- if the classes are already merged in the BN input, no affine
# can split them and the BN direction is dead regardless of labels.
#
# Decisive comparison per condition (dgl_kitti):
#   frozen              : W0 (clean) with frozen BN
#   bn_labeled_affine   : W0 with label-fitted BN affine (gamma/beta)
#   bn_labeled_affine_stats: + re-estimated running stats too
#   ceiling             : W* (pool) with frozen BN
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (500 frames, fog+crosstalk)
#
# Usage:
#   DRY_RUN=1 bash run_probe_bn_labeled.sh 2
#   SMOKE=1   bash run_probe_bn_labeled.sh 2
#   bash run_probe_bn_labeled.sh 2
#   BN_SCOPE=bottleneck bash run_probe_bn_labeled.sh 2
#
# Output: robust_diagnostic/logs/probe_bn_labeled.json

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
echo "Labeled BN probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS, scope=$BN_SCOPE)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_bn_labeled_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --bn_scope $BN_SCOPE --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_bn_labeled.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_bn_labeled.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== LABELED BN OK ==="
  echo "  bn_labeled_affine_delta: does a label-fitted BN affine recover fog/crosstalk?"
  echo "  pre_bn_sep_clean/corr: if corr << clean, classes are merged BEFORE the BN,"
  echo "  so no affine can help -> the BN direction is dead even with labels."
else
  echo "=== LABELED BN FAILED (exit $RC) ==="
  exit $RC
fi
