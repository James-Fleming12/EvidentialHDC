#!/usr/bin/env bash
# run_al_stable_update.sh: C30 stable-update diagnostic (your points 1-20, families A-D).
# All eval-only on COV-SHIFT ep10, k=8 (56 labels) as the cheap budget. Tests whether
# the update can be made small and stable, not whether more labels help.
#
# Usage:
#   bash run_al_stable_update.sh 3                         # ep10, all conds

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [stable-update] ep10 [$CONDS] on $METHOD (k=8, r=8 oracle U) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_stable_update_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "stable_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_stable_update_ep10.json" \
  2>&1 | tee "logs/al_stable_update_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== STABLE-UPDATE OK ==="
  echo "Check logs/al_stable_update_ep10.log:"
  echo "  - A eta/gamma/rank/clip: which stabilization makes r=8 positive?"
  echo "  - B soft vs one-hot: does Y-P0 beat Y-XW0?"
  echo "  - D weight: does uncertainty weighting help?"
else
  echo "=== STABLE-UPDATE FAILED (exit $RC) ==="
  exit $RC
fi
