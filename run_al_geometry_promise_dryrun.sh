#!/usr/bin/env bash
# Dry run of the geometry-promise diagnostic: fast smoke test before the
# overnight. Tiny frames / pool, ONE condition (fog), ep10 only, fewer repeats
# -- every section (A/B/C/D/E/F, synthesis) executes.
#
# Usage:
#   bash run_al_geometry_promise_dryrun.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DRY_OUT="robust_diagnostic/logs/al_geometry_promise_DRYRUN.json"

echo "=== [al_geometry_promise DRYRUN] $CONDS: tiny frames/pool, ep10 only ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_promise_diag.py \
  --path_b "$EP10_CKPT" --method_b "$METHOD" --label "covshift_ep10_dry" \
  --conds "$CONDS" \
  --frames 2 --pool_size 2000 --max_clean 10000 \
  --repeats 3 \
  --out "$DRY_OUT" \
  2>&1 | tee "logs/al_geometry_promise_dryrun.log"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
  echo "=== DRY RUN OK ==="
  echo "Full log: logs/al_geometry_promise_dryrun.log"
  echo "JSON: $DRY_OUT"
  echo "Then launch the overnight: bash run_al_geometry_promise.sh 3"
else
  echo "=== DRY RUN FAILED (exit $RC) ==="
  exit 1
fi
