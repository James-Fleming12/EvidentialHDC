#!/usr/bin/env bash
# run_probe_ingate_stats.sh: micro test of the SELF-DETECTING input-IN gate.
#
# The gate uses per-scan {range, remission} statistics vs a clean reference to
# decide whether to engage input-IN -- NO corruption-type knowledge needed.
# Outputs frozen mIoU per condition for always_off (gate fully off) and several
# tau thresholds (engage iff the scan stats deviate from clean by > tau).
# Compare tau_x vs always_off:
#   * fog/crosstalk: does the gate keep the cov-shift rescue (frozen near the
#     always-on reference)?
#   * snow/wet_ground: does the gate recover healthy capacity (frozen above
#     always-on, toward the plain-DGLSS++ level)?
#
# Usage:
#   bash run_probe_ingate_stats.sh 2
#   MAX_FRAMES=100 CONDS=fog,snow bash run_probe_ingate_stats.sh 2
#
# Output: robust_diagnostic/logs/probe_ingate_stats.json

set -u
set -o pipefail
GPU="${1:-2}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
TAUS="${TAUS:-0.1,0.5,1.0,2.0}"
EXTRACTORS="${EXTRACTORS:-cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES, conds=$CONDS, taus=$TAUS)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_ingate_stats_diag.py \
  --max_frames "$MAX_FRAMES" --conds "$CONDS" --taus "$TAUS" --extractors "$EXTRACTORS" \
  --out "robust_diagnostic/logs/probe_ingate_stats.json" \
  2>&1 | tee "logs/probe_ingate_stats.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== INGATE STATS OK ==="
  echo "  tau_x vs always_off on fog/crosstalk (rescue kept?) and snow/wet (capacity recovered?)"
else
  echo "=== INGATE STATS FAILED (exit $RC) ==="
  exit $RC
fi
