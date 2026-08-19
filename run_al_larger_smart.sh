#!/usr/bin/env bash
# run_al_larger_smart.sh: larger dataset (more varied points + higher gaps) and
# smarter 500 allocation (diversity, info gain) to not starve minority classes.
# All eval-only on COV-SHIFT ep10, 4 conditions. Keeps extractor frozen.
#
# Usage:
#   bash run_al_larger_smart.sh 3                         # ep10, all conds

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"

echo "=== [larger+smart500] ep10 [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_larger_smart_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "larger_smart_ep10" --conds "$CONDS" \
  --out "robust_diagnostic/logs/al_larger_smart_ep10.json" \
  2>&1 | tee "logs/al_larger_smart_ep10.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== LARGER+SMART500 OK ==="
  echo "Check logs/al_larger_smart_ep10.log:"
  echo "  - larger dataset: frozen/oracle/gap small->large (does ceiling rise?)"
  echo "  - smart500: random vs uniform vs diverse vs uncertainty bank mIoU and W_pseudo delta"
else
  echo "=== LARGER+SMART500 FAILED (exit $RC) ==="
  exit $RC
fi
