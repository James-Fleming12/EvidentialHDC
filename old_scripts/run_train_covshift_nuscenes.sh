#!/usr/bin/env bash
# run_train_covshift_nuscenes.sh: train the COV-SHIFT feature extractor on
# NuScenes (KITTI format, 32-beam projection) for later NuScenes-C evaluation.
#
# Same cov-shift recipe as the KITTI extractor (supcon_vib_dglsspp_inputin_in_chan:
# per-scan channels {0,4} normalization + internal InstanceNorm + GMSIFC/LSCC, no
# SupCon), trained on the NuScenes train split (~700 scenes, 21 epochs / 100%).
#
# Usage:
#   bash run_train_covshift_nuscenes.sh 3          # full ~overnight run
#   EPOCHS=2 CUTOFF=0.05 bash run_train_covshift_nuscenes.sh 3   # smoke test
#
# Output: robust_diagnostic/logs/nusc_covshift_21ep/SENet* (per-epoch checkpoints)

set -u
set -o pipefail
GPU="${1:-3}"
EPOCHS="${EPOCHS:-21}"
CUTOFF="${CUTOFF:-1.0}"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/nusc_covshift_21ep}"
echo "Using GPU $GPU ($EPOCHS epochs / $CUTOFF cutoff, log_dir=$LOG_DIR)"

echo "=== [train cov-shift on NuScenes] ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/train_covshift_nuscenes.py \
  --epochs "$EPOCHS" --cutoff "$CUTOFF" --log_dir "$LOG_DIR" \
  2>&1 | tee "logs/train_covshift_nuscenes.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== TRAIN NUSCENES OK ==="
  echo "Checkpoints in $LOG_DIR/SENet*"
  echo "Next: point the full-dataset diag's --extractors at $LOG_DIR and evaluate on NuScenes-C."
else
  echo "=== TRAIN NUSCENES FAILED (exit $RC) ==="
  exit $RC
fi
