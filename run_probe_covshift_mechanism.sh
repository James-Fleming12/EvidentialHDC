#!/usr/bin/env bash
# run_probe_covshift_mechanism.sh: Iteration-0 mechanism diagnostics
# (docs/cov_shift/cov_full_scale.md) -- eval-only, on the 4 existing checkpoints:
#
#   D1 clean baseline decomposition (frozen + ceiling on clean)
#   D2 per-class recoverable map per condition
#   D3 input-statistics calibration (per-scan {0,4} mean/var)
#   D4 residual + conditioning + lambda sweep {1e-4,1e-3,1e-2}
#   D5 code-vs-raw separability hooks + bit balance + sign margin
#   D6 variance / effective-rank of code and features
#   D7 normalization-lever ablation (model.input_in disabled at eval, gate test)
#   D9 W0-source control (nuScenes-clean fit) for the NuScenes extractors
#   D10 R1-vs-R4 headroom (proto_ceiling vs linear_ceiling)
#
# Usage:
#   bash run_probe_covshift_mechanism.sh 3
#   MAX_FRAMES=100 bash run_probe_covshift_mechanism.sh 3   # smoke test
#   EXTRACTORS=cov_kitti,dgl_kitti bash run_probe_covshift_mechanism.sh 3
#
# Output: robust_diagnostic/logs/probe_covshift_mechanism_ep10.json

set -u
set -o pipefail
GPU="${1:-3}"
MAX_FRAMES="${MAX_FRAMES:-0}"
EXTRACTORS="${EXTRACTORS:-all}"
GATE_OFF="${GATE_OFF:-1}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES, extractors=$EXTRACTORS, gate_off=$GATE_OFF)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_covshift_mechanism_diag.py \
  --max_frames "$MAX_FRAMES" --extractors "$EXTRACTORS" --gate_off "$GATE_OFF" \
  --out "robust_diagnostic/logs/probe_covshift_mechanism_ep10.json" \
  2>&1 | tee "logs/probe_covshift_mechanism_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== COVSHIFT MECHANISM OK ==="
  echo "Check robust_diagnostic/logs/probe_covshift_mechanism_ep10.json"
  echo "  D1 clean: interaction term per condition (healthy deficit split)"
  echo "  D4: effrank_pr + lambda sweep (ceiling cap vs conditioning)"
  echo "  D7: ceiling_gate_off - ceiling (does gating off normalization recover headroom?)"
  echo "  D9: frozen_W0_alt vs frozen (probe-source interaction)"
else
  echo "=== COVSHIFT MECHANISM FAILED (exit $RC) ==="
  exit $RC
fi
