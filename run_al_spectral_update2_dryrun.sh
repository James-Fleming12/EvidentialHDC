#!/usr/bin/env bash
# Dry run of the Iteration-10 spectral update2 diagnostic: fast smoke test
# before the overnight. Tiny frames / pool, ONE condition (fog), ep10 only,
# small sweeps -- every section (normalized spectrum, validation, 9A/9B/9E,
# 10-COMB, budget, synthesis) executes.
#
# Usage:
#   bash run_al_spectral_update2_dryrun.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DRY_OUT="robust_diagnostic/logs/al_spectral_update2_DRYRUN.json"

echo "=== [al_spectral_update2 DRYRUN] $CONDS: tiny frames/pool, ep10 only ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_spectral_update2_diag.py \
  --path_b "$EP10_CKPT" --method_b "$METHOD" --label "covshift_ep10_dry" \
  --conds "$CONDS" \
  --frames 2 --pool_size 2000 --max_clean 10000 \
  --betas "0.0,0.5,1.0" --combo_betas "0.25,0.5" \
  --gammas "1,4" --etas "0.2,0.8" --drop_ps "10,40" \
  --mean_ks "8,32" \
  --out "$DRY_OUT" \
  2>&1 | tee "logs/al_spectral_update2_dryrun.log"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
  echo "=== DRY RUN OK ==="
  echo "Full log: logs/al_spectral_update2_dryrun.log"
  echo "JSON: $DRY_OUT"
  echo "Then launch the overnight: bash run_al_spectral_update2.sh 3"
else
  echo "=== DRY RUN FAILED (exit $RC) ==="
  exit 1
fi
