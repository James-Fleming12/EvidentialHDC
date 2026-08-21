#!/usr/bin/env bash
# run_overnight.sh: overnight batch -- three diagnostics/training runs on one GPU.
#
#   1. run_probe_projection.sh   HDC projection variants for linear separability
#                                and mIoU (eval-only, frozen cov-shift ep10).
#   2. run_probe_al_gauge.sh     can a naive label-free signal predict whether a
#                                condition is worth active learning? (eval-only).
#   3. run_train_covshift_nuscenes.sh  train the cov-shift extractor on NuScenes
#                                for later NuScenes-C evaluation.
#
# Runs 1-2 first (fast, ~1-2h total), then 3 (training, ~2-4h), so the eval
# diagnostics complete even if the training is interrupted.
#
# Usage:
#   bash run_overnight.sh 3                    # full overnight run on GPU 3
#   DRY_RUN=TRUE bash run_overnight.sh 3       # tiny smoke: 2 frames, 1 epoch,
#                                              # 1 condition, 2 projection variants
#
# Config via env (defaults are the full runs):
#   GPU, DRY_RUN, CONDS, MAX_FRAMES, EPOCHS, CUTOFF, VARIANTS

set -u
set -o pipefail
GPU="${1:-3}"
DRY_RUN="${DRY_RUN:-FALSE}"

if [ "$DRY_RUN" = "TRUE" ]; then
  echo "=== DRY RUN: tiny smoke settings ==="
  CONDS="${CONDS:-fog}"
  MAX_FRAMES="${MAX_FRAMES:-2}"
  EPOCHS="${EPOCHS:-1}"
  CUTOFF="${CUTOFF:-0.01}"
  VARIANTS="${VARIANTS:-bern,gauss}"
  TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-robust_diagnostic/logs/nusc_covshift_smoke}"
else
  CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
  MAX_FRAMES="${MAX_FRAMES:-0}"
  EPOCHS="${EPOCHS:-21}"
  CUTOFF="${CUTOFF:-1.0}"
  VARIANTS="${VARIANTS:-bern,gauss,sparse_k1,sparse_k8,ternary,zca,within_whn,rotated,dim5k,dim20k,concat2}"
  TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-robust_diagnostic/logs/nusc_covshift_21ep}"
fi
echo "Using GPU $GPU (dry_run=$DRY_RUN, conds=$CONDS, max_frames=$MAX_FRAMES)"
echo "  train: ${EPOCHS} ep / ${CUTOFF} cutoff -> $TRAIN_LOG_DIR"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# ---- [1/3] projection variants (eval-only) ----
echo ""
echo "=== [1/3] probe_projection (variants=$VARIANTS) ==="
CONDS="$CONDS" MAX_FRAMES="$MAX_FRAMES" VARIANTS="$VARIANTS" \
  bash run_probe_projection.sh "$GPU" || fail "projection"

# ---- [2/3] label-free AL gauge (eval-only) ----
echo ""
echo "=== [2/3] probe_al_gauge ==="
CONDS="$CONDS" MAX_FRAMES="$MAX_FRAMES" \
  bash run_probe_al_gauge.sh "$GPU" || fail "al-gauge"

# ---- [3/3] train cov-shift on NuScenes ----
echo ""
echo "=== [3/3] train cov-shift on NuScenes ($EPOCHS ep / $CUTOFF cutoff) ==="
EPOCHS="$EPOCHS" CUTOFF="$CUTOFF" LOG_DIR="$TRAIN_LOG_DIR" \
  bash run_train_covshift_nuscenes.sh "$GPU" || fail "train-nuscenes"

echo ""
echo "=== OVERNIGHT COMPLETE ==="
echo "  projection:  robust_diagnostic/logs/probe_projection_ep10.json"
echo "  al gauge:    robust_diagnostic/logs/probe_al_gauge_ep10.json"
echo "  nusc ckpt:   $TRAIN_LOG_DIR/SENet*"
