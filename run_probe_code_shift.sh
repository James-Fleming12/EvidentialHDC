#!/usr/bin/env bash
# run_probe_code_shift.sh: what actually separates the fog/crosstalk WINNER
# (cov-shift, input-IN) from the equal-raw-shift LOSER (GeoID) -- is it the
# CODE space, not the raw feature space?
#
# The GeoID-loss extractor matched cov's RAW 128-d class_shift but did NOT get
# the zero-shot/ceiling. This probe measures shift + fisher + W0-alignment in
# the BINARIZED 10000-d CODE space (where W0 operates), plus an ablation that
# applies cov-style input-IN at eval to the GeoID model.
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (200 frames, fog+crosstalk)
#
# Usage:
#   DRY_RUN=1 bash run_probe_code_shift.sh 2
#   SMOKE=1   bash run_probe_code_shift.sh 2
#   bash run_probe_code_shift.sh 2
#
# Output: robust_diagnostic/logs/probe_code_shift.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-hyper_kitti:baseline:logs/kitti_pretrain,cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan,geoid:supcon_vib_geoid:robust_diagnostic/logs/geoid_full/supcon_vib_geoid}"
echo "Code-shift probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_code_shift_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_code_shift.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD"
RC=$?
echo "  [exit code: $RC]"

if [ $RC -eq 0 ]; then
  echo "=== CODE-SHIFT OK ==="
  echo "  If cov's code_shift << geoid's code_shift (despite equal raw_shift), the"
  echo "  real lever is CODE-space stability, not raw-feature stability."
  echo "  If geoid_inputin_eval_frozen jumps toward cov, the input-IN eval transform"
  echo "  (not the learned weights) is what makes W0 transfer."
else
  echo "=== CODE-SHIFT FAILED (exit $RC) ==="
  exit $RC
fi
