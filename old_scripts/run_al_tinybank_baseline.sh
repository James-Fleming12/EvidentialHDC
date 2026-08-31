#!/usr/bin/env bash
# run_al_tinybank_baseline.sh: tiny bank (500) as new baseline if it helps every cond.
# All eval-only on COV-SHIFT ep10, 4 conditions.
#
# Usage:
#   bash run_al_tinybank_baseline.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [tinybank-baseline] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_tinybank_baseline_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "tinybank_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_tinybank_baseline_ep10.json" \
  2>&1 | tee "logs/al_tinybank_baseline_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== TINYBANK-BASELINE OK ==="
  echo "Check logs/al_tinybank_baseline_ep10.log:"
  echo "  - tiny 1-NN (56 vs 556/1056/5056) vs frozen on every cond"
  echo "  - tiny+ pseudo (500 pseudo) vs true 500 (oracle for those 500)"
else
  echo "=== TINYBANK-BASELINE FAILED (exit $RC) ==="
  exit $RC
fi
