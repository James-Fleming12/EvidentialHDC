#!/usr/bin/env bash
# Full medium run (21 ep / 100%, ~10h) for the covariate-shift-aware DGLSS++
# candidate. Defaults to the Iteration-19.12 winner (channel-restricted input-IN);
# pass a different method to run the scale-only variant instead if the micro favors it.
#
# Usage:
#   bash run_covshift_medium.sh 3                  # GPU 3, channel-restricted (19.12)
#   bash run_covshift_medium.sh 3 supcon_vib_dglsspp_inputin_in_scale   # scale-only
#   bash run_covshift_medium.sh 3 <method> 21      # custom epochs

set -u
GPU="${1:-3}"
METHOD="${2:-supcon_vib_dglsspp_inputin_in_chan}"
EPOCHS="${3:-21}"
echo "Using GPU $GPU, method $METHOD, $EPOCHS ep / 100%"

CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$METHOD" --epochs "$EPOCHS" --cutoff 1.0 \
  --log_dir "robust_diagnostic/logs/med_$METHOD" \
  2>&1 | tee "logs/covshift_med_train.log"

echo "Done. Next: the full battery vs DGLSS++ and Robust (extractor_diff, tta_ceiling,"
echo "frozen_ceiling) on the med_$METHOD checkpoint."
