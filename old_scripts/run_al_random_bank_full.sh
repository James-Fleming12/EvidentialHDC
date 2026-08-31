#!/usr/bin/env bash
# run_al_random_bank_full.sh: random 500-point bank baseline on FULL dataset (8 conditions).
# One run produces BOTH the README-accurate reference columns (zero-shot W0 = frozen,
# ceiling W* = oracle, per condition) AND the random-bank AL numbers (1-NN, W_res
# pseudo/true at 56+500 labels) on the README R4 harness: 100 frames, 100k pool / 100k
# val, seed-42 split, spectral-exact ridge solve, and a 200k-point clean W0 fit (the
# bigger clean fit is the accurate zero-shot: fog ~0.277 vs the README's under-fit
# ~0.235; raise CLEAN_FIT to 400k to check saturation).
#
# Usage:
#   bash run_al_random_bank_full.sh 3            # clean fit 200k
#   CLEAN_FIT=400000 bash run_al_random_bank_full.sh 3
#
# Output: robust_diagnostic/logs/al_random_bank_full_ep10_readme.json with, per cond:
#   refs.frozen_small / refs.oracle_small / refs.gap_small   (zs + ceiling + gap)
#   bank.W_res_pseudo_delta_small / bank.W_res_true_small    (AL numbers)

set -u
set -o pipefail
GPU="${1:-3}"
CLEAN_FIT="${CLEAN_FIT:-200000}"
echo "Using GPU $GPU (clean W0 fit on $CLEAN_FIT points)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [random-bank-full] ep10 [8 conds, README R4 harness + exact solve + ${CLEAN_FIT}k clean fit] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_random_bank_full_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "random_full_ep10_readme" \
  --clean_fit_n "$CLEAN_FIT" \
  --out "robust_diagnostic/logs/al_random_bank_full_ep10_readme.json" \
  2>&1 | tee "logs/al_random_bank_full_ep10_readme.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== RANDOM-BANK-FULL OK ==="
  echo "Check logs/al_random_bank_full_ep10_readme.log:"
  echo "  - frozen (zero-shot W0) vs oracle (ceiling W*) gap per condition"
  echo "  - bank 1-NN vs W_res pseudo delta at 56+500 random (the AL numbers)"
else
  echo "=== RANDOM-BANK-FULL FAILED (exit $RC) ==="
  exit $RC
fi
