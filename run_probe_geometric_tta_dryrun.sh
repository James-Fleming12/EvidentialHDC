#!/usr/bin/env bash
# Dry run of the geometric-TTA diagnostic: fast smoke test before the overnight.
# Catches import / CLI / data-path / checkpoint / solver errors in minutes.
# Tiny frames / pool / val, ONE condition (fog), ep10 only -- every section
# (A procrustes, B coral, C diffusion, controls, synthesis) executes.
#
# Usage:
#   bash run_probe_geometric_tta_dryrun.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DRY_OUT="robust_diagnostic/logs/probe_geometric_tta_DRYRUN.json"

echo "=== [geometric_tta DRYRUN] $CONDS: tiny frames/pool/val, ep10 only ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_geometric_tta_diag.py \
  --path_b "$EP10_CKPT" --method_b "$METHOD" --label "covshift_ep10_dry" \
  --conds "$CONDS" \
  --frames 2 --pool_size 2000 --val_size 2000 --max_clean 10000 \
  --out "$DRY_OUT" \
  2>&1 | tee "logs/probe_geometric_tta_dryrun.log"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
  echo "=== DRY RUN OK ==="
  echo "Full log: logs/probe_geometric_tta_dryrun.log"
  echo "JSON: $DRY_OUT"
  echo "Then launch the overnight: bash run_probe_geometric_tta.sh 3"
else
  echo "=== DRY RUN FAILED (exit $RC) ==="
  exit 1
fi
