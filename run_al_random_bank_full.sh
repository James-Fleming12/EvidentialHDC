#!/usr/bin/env bash
# run_al_random_bank_full.sh: random 500-point bank baseline on FULL dataset (8 conditions, larger pool).
# All eval-only on COV-SHIFT ep10. This is the baseline the next improvements will beat.
#
# Usage:
#   bash run_al_random_bank_full.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
echo "Using GPU $GPU"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [random-bank-full] ep10 [all 8 conds, 100 vs 200 frames] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_random_bank_full_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "random_full_ep10" \
  --out "robust_diagnostic/logs/al_random_bank_full_ep10.json" \
  2>&1 | tee "logs/al_random_bank_full_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== RANDOM-BANK-FULL OK ==="
  echo "Check logs/al_random_bank_full_ep10.log:"
  echo "  - small vs large gap (does larger dataset raise ceiling?)"
  echo "  - bank 1-NN vs W_res pseudo delta at 56+500 random"
else
  echo "=== RANDOM-BANK-FULL FAILED (exit $RC) ==="
  exit $RC
fi
