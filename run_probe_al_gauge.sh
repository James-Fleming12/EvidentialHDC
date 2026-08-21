#!/usr/bin/env bash
# run_probe_al_gauge.sh: label-free AL gauge diagnostic on the frozen cov-shift
# ep10 extractor. Tests whether a naive label-free signal (norm inflation, mean
# shift, probe confidence drop, code dead-frac/Hamming, R4-vs-R1 disagreement)
# predicts the measured AL-closeable gap across conditions (Spearman rho).
#
# Usage:
#   bash run_probe_al_gauge.sh 3
#   CONDS=fog,wet_ground bash run_probe_al_gauge.sh 3
#   MAX_FRAMES=200 bash run_probe_al_gauge.sh 3   # smoke test
#
# Output: robust_diagnostic/logs/probe_al_gauge_ep10.json
#   per condition: label-free signals + measured gap + per-signal Spearman rho

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
SUFFIX="ep10"
[ "$MAX_FRAMES" != "0" ] && SUFFIX="${SUFFIX}_f${MAX_FRAMES}"

echo "=== [AL gauge diag] $SUFFIX [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_al_gauge_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "gauge_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/probe_al_gauge_${SUFFIX}.json" \
  2>&1 | tee "logs/probe_al_gauge_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== AL GAUGE OK ==="
  echo "Check logs/probe_al_gauge_${SUFFIX}.log:"
  echo "  - per-condition label-free signals vs the measured closeable gap"
  echo "  - Spearman rho(gap, signal) + the best single gauge signal"
else
  echo "=== AL GAUGE FAILED (exit $RC) ==="
  exit $RC
fi
