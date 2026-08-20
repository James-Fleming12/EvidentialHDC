#!/usr/bin/env bash
# run_al_full_dataset.sh: PAPER-READY full-dataset mIoU on EVERY point of EVERY
# frame of KITTI seq 08, per corrupted condition, for BOTH extractors and BOTH
# decoders:
#   * cov-shift ep10 (supcon_vib_dglsspp_inputin_in_chan):
#       - R4 linear: zero-shot W0 / ceiling W* / AL W_res (56+500 random bank)
#       - R1 prototype: clean protos (zs) / corrupted-pool protos (ceiling)
#   * DGLSS++ base (supcon_vib_dglsspp): same R4 + R1 zero-shot vs ceiling
#   * Robust DGLSS++ (supcon_vib_dglsspp_corsupcon, med 21ep): R4 + R1, for the
#     README only (not in the official paper tables)
#
# Same probe machinery as the README harness (exact ridge, 200k clean fit,
# 400k pool, oracle U r=8) but the VAL set is the full dataset (~4k frames
# x ~130k pts/scan per condition) instead of 100 frames. Memory-bounded:
# reservoir pool + streaming confusion accumulation.
#
# Usage:
#   bash run_al_full_dataset.sh 3                # both extractors, all 8 conds
#   CONDS=fog,crosstalk bash run_al_full_dataset.sh 3   # subset
#   MAX_FRAMES=200 bash run_al_full_dataset.sh 3        # quick smoke test
#
# Output: robust_diagnostic/logs/al_full_dataset_ep10.json
#   extractors[<label>].conds[<cond>] = {
#     linear_frozen / linear_ceiling / linear_gap / linear_W_res_pseudo(+delta) /
#     linear_W_res_true(+delta) / proto_frozen / proto_ceiling / proto_gap / n_val }

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
MAX_FRAMES="${MAX_FRAMES:-0}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
ROBUST_CKPT="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
SUFFIX="ep10"
[ "$MAX_FRAMES" != "0" ] && SUFFIX="${SUFFIX}_f${MAX_FRAMES}"

echo "=== [full-dataset] $SUFFIX [$CONDS] on cov-shift ep10 + DGLSS++ + Robust DGLSS++ ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_full_dataset_diag.py \
  --label "full_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --extractors \
    "cov_ep10:${METHOD}:${CKPT},dglsspp:supcon_vib_dglsspp:${DGLSSPP_CKPT},robust:supcon_vib_dglsspp_corsupcon:${ROBUST_CKPT}" \
  --out "robust_diagnostic/logs/al_full_dataset_${SUFFIX}.json" \
  2>&1 | tee "logs/al_full_dataset_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== FULL-DATASET OK ==="
  echo "Check logs/al_full_dataset_${SUFFIX}.log:"
  echo "  - [R4] frozen / ceiling / W_res pseudo+true on the FULL val set"
  echo "  - [R1] prototype frozen / ceiling on the FULL val set"
  echo "  - for cov-shift ep10, DGLSS++, and Robust DGLSS++ extractors"
else
  echo "=== FULL-DATASET FAILED (exit $RC) ==="
  exit $RC
fi
