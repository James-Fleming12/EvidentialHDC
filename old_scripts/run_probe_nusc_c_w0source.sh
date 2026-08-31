#!/usr/bin/env bash
# run_probe_nusc_c_w0source.sh: validate the corrected NuScenes-C story with one
# consistent run -- zero-shot W0 fit on nuScenes-clean (in-domain) instead of
# KITTI-clean, ceiling + W_res + full val via the authoritative eval_target_condition.
#
# Usage:
#   bash run_probe_nusc_c_w0source.sh 2
#   MAX_FRAMES=100 bash run_probe_nusc_c_w0source.sh 2   # smoke
#   SKIP_EXISTING=1 bash run_probe_nusc_c_w0source.sh 2   # resume
#
# Output: robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json
#   extractors[<label>].conds[<cond>] = { linear_frozen (in-domain W0),
#     linear_ceiling, linear_gap, linear_W_res_*, frozen_kittiW0 (reference),
#     delta_zs_in_vs_cross }

set -u
set -o pipefail
GPU="${1:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES, skip_existing=$SKIP_EXISTING)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py \
  --max_frames "$MAX_FRAMES" --skip_existing "$SKIP_EXISTING" \
  --out "robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json" \
  2>&1 | tee "logs/probe_nusc_c_w0source_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== NUSC W0-SOURCE OK ==="
  echo "  linear_frozen      : zero-shot with IN-DOMAIN nuScenes-clean W0 (the corrected number)"
  echo "  linear_ceiling     : ceiling (in-domain corrupted pool)"
  echo "  frozen_kittiW0     : reference frozen with KITTI-clean W0 (contaminated baseline)"
  echo "  delta_zs_in_vs_cross : in-domain minus cross-domain zero-shot"
else
  echo "=== NUSC W0-SOURCE FAILED (exit $RC) ==="
  exit $RC
fi
