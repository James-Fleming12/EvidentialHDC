#!/usr/bin/env bash
# run_al_full_dataset.sh: PAPER-READY full-dataset mIoU (zero-shot W0, ceiling W*,
# and the 56+500 random-bank AL W_res) on EVERY point of EVERY frame of KITTI
# seq 08, per corrupted condition. Streaming/reservoir harness, memory-bounded.
#
# Same probe machinery as the README R4 harness (exact ridge, 200k clean fit,
# 100k pool, oracle U r=8) but the VAL set is the full dataset (~4k frames
# x ~130k pts/scan per condition) instead of 100 frames.
#
# Usage:
#   bash run_al_full_dataset.sh 3                # all 8 conditions, one GPU
#   CONDS=fog,crosstalk bash run_al_full_dataset.sh 3   # subset
#   MAX_FRAMES=200 bash run_al_full_dataset.sh 3        # quick smoke test
#
# Output: robust_diagnostic/logs/al_full_dataset_ep10.json with, per cond:
#   frozen / ceiling / gap / W_res_pseudo (+delta) / W_res_true (+delta) / n_val

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
SUFFIX="ep10"
[ "$MAX_FRAMES" != "0" ] && SUFFIX="${SUFFIX}_f${MAX_FRAMES}"

echo "=== [full-dataset] $SUFFIX [$CONDS] on $METHOD ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_full_dataset_diag.py \
  --path_b "$CKPT" --method_b "$METHOD" --label "full_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --out "robust_diagnostic/logs/al_full_dataset_${SUFFIX}.json" \
  2>&1 | tee "logs/al_full_dataset_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== FULL-DATASET OK ==="
  echo "Check logs/al_full_dataset_${SUFFIX}.log:"
  echo "  - frozen (zero-shot W0) / ceiling (W*) / W_res pseudo+true on the FULL val set"
  echo "  - closeable gap and % closed at the full-dataset scale"
else
  echo "=== FULL-DATASET FAILED (exit $RC) ==="
  exit $RC
fi
