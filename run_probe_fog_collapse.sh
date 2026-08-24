#!/usr/bin/env bash
# run_probe_fog_collapse.sh: is KITTI-C fog so different from NuScenes-C fog that
# DGLSS++'s feature space collapses on one but not the other?
#
# Direct test per (extractor, dataset, condition):
#   * mean per-class nearest-mean recall in the raw 128-d features AND the
#     binarized code on the CORRUPTED stream (clean prototypes, in-domain W0).
#     If the space collapses, recall -> the ~1/17 random baseline.
#   * frozen R4 mIoU with an in-domain W0 (the corrected zero-shot).
#
# Defaults: DGLSS++ + cov-shift (KITTI pair), fog + crosstalk, 200 frames.
# Fast: 2 extractors x 2 datasets x 2 conds, capped frames.
#
# Usage:
#   bash run_probe_fog_collapse.sh 2
#   EXTRACTORS=cov_kitti,... CONDS=fog bash run_probe_fog_collapse.sh 2
#
# Output: robust_diagnostic/logs/probe_fog_collapse.json

set -u
set -o pipefail
GPU="${1:-2}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk}"
EXTRACTORS="${EXTRACTORS:-dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES, conds=$CONDS)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_fog_collapse_diag.py \
  --max_frames "$MAX_FRAMES" --conds "$CONDS" --extractors "$EXTRACTORS" \
  --out "robust_diagnostic/logs/probe_fog_collapse.json" \
  2>&1 | tee "logs/probe_fog_collapse.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== FOG COLLAPSE OK ==="
  echo "  mean_recall_code/feat: KITTI-C fog should COLLAPSE (~0.1-0.2), NuScenes-C fog should NOT (~0.4+)"
else
  echo "=== FOG COLLAPSE FAILED (exit $RC) ==="
  exit $RC
fi
