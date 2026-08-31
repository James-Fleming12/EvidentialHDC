#!/usr/bin/env bash
# run_al_lowbonus_bank.sh: why snow/crosstalk bonuses low, U without oracle, tiny bank.
# All eval-only on COV-SHIFT ep10, 4 conditions. Keeps extractor frozen.
#
# Usage:
#   bash run_al_lowbonus_bank.sh 3                         # ep10, all conds

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [lowbonus+bank] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_lowbonus_bank_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "lowbonus_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_lowbonus_bank_ep10.json" \
  2>&1 | tee "logs/al_lowbonus_bank_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== LOWBONUS+BANK OK ==="
  echo "Check logs/al_lowbonus_bank_ep10.log:"
  echo "  - Q1 low bonuses: t_cos vs w_cos vs B1 at k=32 (what AL framework misses)"
  echo "  - Q2 U without oracle: U_T / U_Rreg / U_pool / U_shift align and delta"
  echo "  - Q3 tiny bank: 1-NN acc vs bank size (56 vs 556/1056/5056)"
else
  echo "=== LOWBONUS+BANK FAILED (exit $RC) ==="
  exit $RC
fi
