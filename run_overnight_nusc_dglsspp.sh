#!/usr/bin/env bash
# run_overnight_nusc_dglsspp.sh: train the BASE DGLSS++ extractor (no cov-shift
# normalization) on NuScenes, then evaluate it on NuScenes-C zero-shot + ceiling.
#
# Pipeline:
#   1. Train supcon_vib_dglsspp on the NuScenes train split (KITTI format,
#      32-beam projection, 21 epochs / 100%) ->
#      robust_diagnostic/logs/nusc_dglsspp_21ep/SENet*
#      (the same recipe as the KITTI DGLSS++ reference, now on NuScenes data)
#   2. Full-dataset NuScenes-C eval of that checkpoint (frozen W0 / ceiling W*,
#      R4 + R1) over all 8 conditions x heavy severity ->
#      robust_diagnostic/logs/al_nuscenes_c_dglsspp.json
#
# This gives the same-domain baseline to compare against the cov-shift NuScenes
# run (nusc_covshift_21ep -> al_nuscenes_c.json): both trained on NuScenes, both
# evaluated on NuScenes-C, differing only in the cov-shift normalization.
#
# Usage:
#   bash run_overnight_nusc_dglsspp.sh 3               # full ~overnight run
#   EPOCHS=2 CUTOFF=0.05 bash run_overnight_nusc_dglsspp.sh 3   # smoke test
#
# Outputs:
#   robust_diagnostic/logs/nusc_dglsspp_21ep/SENet*
#   robust_diagnostic/logs/al_nuscenes_c_dglsspp.json

set -u
set -o pipefail
GPU="${1:-3}"
EPOCHS="${EPOCHS:-21}"
CUTOFF="${CUTOFF:-1.0}"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/nusc_dglsspp_21ep}"
SEVS="${SEVS:-heavy}"
echo "Using GPU $GPU (epochs=$EPOCHS, cutoff=$CUTOFF, log_dir=$LOG_DIR, sevs=$SEVS)"

# --- 1. train DGLSS++ (base) on NuScenes ---
echo ""
echo "=== [train DGLSS++ on NuScenes] ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/train_covshift_nuscenes.py \
  --epochs "$EPOCHS" --cutoff "$CUTOFF" --log_dir "$LOG_DIR" \
  --method supcon_vib_dglsspp \
  2>&1 | tee "logs/train_nusc_dglsspp.log"
RC=$?
if [ $RC -ne 0 ]; then
  echo "=== TRAIN NUSCENES DGLSS++ FAILED (exit $RC) ==="
  exit $RC
fi
echo "=== TRAIN OK: checkpoints in $LOG_DIR/SENet* ==="

# --- 2. evaluate that checkpoint on NuScenes-C (zero-shot + ceiling) ---
echo ""
echo "=== [eval DGLSS++ on NuScenes-C] ==="
EXTRACTORS="nusc_dglsspp:supcon_vib_dglsspp:${LOG_DIR}" \
OUT_NAME=al_nuscenes_c_dglsspp \
SEVS="$SEVS" \
CUDA_VISIBLE_DEVICES=$GPU bash run_al_nuscenes_c.sh "$GPU"

echo ""
echo "=== OVERNIGHT DGLSS++ NUSCENES RUN DONE ==="
echo "  checkpoints: $LOG_DIR/SENet*"
echo "  results:     robust_diagnostic/logs/al_nuscenes_c_dglsspp.json"
