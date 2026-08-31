#!/usr/bin/env bash
# Dry run of the T-synthesis diagnostic: fast smoke test before the overnight.
# Tiny frames / pool, ONE condition (fog), ep10 only, small mean-k sweep --
# every section (A sample complexity, B 7A-7F, mass est, synthesis) runs.
#
# Usage:
#   bash run_al_tsynthesis_dryrun.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DRY_OUT="robust_diagnostic/logs/al_tsynthesis_DRYRUN.json"

echo "=== [al_tsynthesis DRYRUN] $CONDS: tiny frames/pool, ep10 only ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_tsynthesis_diag.py \
  --path_b "$EP10_CKPT" --method_b "$METHOD" --label "covshift_ep10_dry" \
  --conds "$CONDS" \
  --frames 2 --pool_size 2000 --max_clean 10000 \
  --labels_per_class 1 --mean_ks "2,8,32" --mean_repeats 3 \
  --out "$DRY_OUT" \
  2>&1 | tee "logs/al_tsynthesis_dryrun.log"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
  echo "=== DRY RUN OK ==="
  echo "Full log: logs/al_tsynthesis_dryrun.log"
  echo "JSON: $DRY_OUT"
  echo "Then launch the overnight: bash run_al_tsynthesis.sh 3"
else
  echo "=== DRY RUN FAILED (exit $RC) ==="
  exit 1
fi
