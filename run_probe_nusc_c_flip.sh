#!/usr/bin/env bash
# run_probe_nusc_c_flip.sh: why does DGLSS++ beat cov-shift on NuScenes-C at the
# CEILING when on KITTI-C cov-shift wins the ceiling 2-3x?
#
# Measures, per extractor / per condition (full harness: 200k clean fit, 400k
# pool, spectral-exact ridge, pool excluded from val):
#   * recoverable residual ||W*-W0||/||W0||   (the ceiling-addable structure)
#   * per-class frozen/ceiling/gap + corrupted-pool per-class support
#   * code-space + raw-feature nearest-mean separability (clean vs pool protos)
#   * code mean-shift / confidence drop (the AL-gauge signals)
#
# Extractors:
#   cov_kitti / dgl_kitti  : KITTI-trained pair on KITTI-C fog,crosstalk,wet_ground
#   cov_nusc  / dgl_nusc   : NuScenes-trained pair on NuScenes-C (all 8 heavy)
#
# Usage:
#   bash run_probe_nusc_c_flip.sh 3
#   MAX_FRAMES=200 bash run_probe_nusc_c_flip.sh 3    # smoke test
#
# Output: robust_diagnostic/logs/probe_nusc_c_flip_ep10.json

set -u
set -o pipefail
GPU="${1:-3}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_nusc_c_dglsspp_vs_covshift_diag.py \
  --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/probe_nusc_c_flip_ep10.json" \
  2>&1 | tee "logs/probe_nusc_c_flip_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== NUSC FLIP PROBE OK ==="
  echo "Check robust_diagnostic/logs/probe_nusc_c_flip_ep10.json"
  echo "  resid_rel: is cov-shift's recoverable residual systematically smaller on NuScenes-C?"
  echo "  per_class_gap / pool_support: which classes drive the DGLSS++ ceiling"
else
  echo "=== NUSC FLIP PROBE FAILED (exit $RC) ==="
  exit $RC
fi
