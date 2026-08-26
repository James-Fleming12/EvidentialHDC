#!/usr/bin/env bash
# run_probe_ceiling_under_mod.sh: do input normalization / BN-stat changes create
# a MORE ROBUST FEATURE SPACE (higher recoverable ceiling), or only re-fit the
# zero-shot classifier?
#
# The earlier BN/input probes (Diagnostics 8, 8b, 9) always fit W* on FROZEN
# features. This probe refits BOTH W0 (clean) and W* (corrupted pool) on the
# MODIFIED features, so ceiling-under-modification is compared to the frozen
# ceiling directly.
#
# Modes (each applied to clean+pool+val feature extraction, then W0+W* refit):
#   none         : baseline frozen ceiling
#   inputin_on   : per-scan input-IN on channels {0,4} on every scan
#   inputin_gate : input-IN only when bn_mismatch_conv_1 > tau (auto-calibrated)
#   bn_reanchor  : re-estimate late-BN running stats on the corrupted stream
#
# Decisive: if a mode's ceiling > the none-mode ceiling, the modification creates
# real recoverable headroom for TTA/AL (the "better ceiling" the paper pivot
# needs for fog/crosstalk); if equal, it only re-fits the classifier.
#
# Usage:
#   DRY_RUN=1 bash run_probe_ceiling_under_mod.sh 2
#   SMOKE=1   bash run_probe_ceiling_under_mod.sh 2
#   bash run_probe_ceiling_under_mod.sh 2
#   MODES=none,inputin_on bash run_probe_ceiling_under_mod.sh 2
#
# Output: robust_diagnostic/logs/probe_ceiling_under_mod.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk}"
MODES="${MODES:-none,inputin_on,inputin_gate,bn_reanchor}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp}"
echo "Ceiling-under-modification probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS, modes=$MODES)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_ceiling_under_mod_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --modes $MODES --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_ceiling_under_mod.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_ceiling_under_mod.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== CEILING-UNDER-MOD OK ==="
  echo "  For each mode, ceiling is fit AND decoded on the MODIFIED features."
  echo "  A mode ceiling > none-mode ceiling = real headroom gain for fog/crosstalk."
else
  echo "=== CEILING-UNDER-MOD FAILED (exit $RC) ==="
  exit $RC
fi
