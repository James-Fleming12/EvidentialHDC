#!/usr/bin/env bash
# Dry run of the dimension-budget diagnostic: fast smoke test before the
# overnight. Tiny frames / pool, ONE condition (fog), ep10 only, fewer dims and
# budgets -- every section (per-dim curve, refs, crossing points, synthesis)
# executes.
#
# Usage:
#   bash run_al_dimension_budget_dryrun.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DRY_OUT="robust_diagnostic/logs/al_dimension_budget_DRYRUN.json"

echo "=== [al_dimension_budget DRYRUN] $CONDS: tiny frames/pool, ep10 only ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_dimension_budget_diag.py \
  --path_b "$EP10_CKPT" --method_b "$METHOD" --label "covshift_ep10_dry" \
  --conds "$CONDS" \
  --frames 2 --pool_size 2000 --max_clean 10000 \
  --dims "128,1000,10000" --budgets "100,500,2000" \
  --out "$DRY_OUT" \
  2>&1 | tee "logs/al_dimension_budget_dryrun.log"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
  echo "=== DRY RUN OK ==="
  echo "Full log: logs/al_dimension_budget_dryrun.log"
  echo "JSON: $DRY_OUT"
  echo "Then launch the overnight: bash run_al_dimension_budget.sh 3"
else
  echo "=== DRY RUN FAILED (exit $RC) ==="
  exit 1
fi
