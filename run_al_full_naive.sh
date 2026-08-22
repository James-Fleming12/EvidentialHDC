#!/usr/bin/env bash
# run_al_full_naive.sh: FULL-DATASET confirmation that the findings that closed
# the naive TTA / naive AL routes still hold at full KITTI-C scale, and that the
# minority-class weakness is/ isn't in the feature space itself.
#
# Same paper-ready harness as run_al_full_dataset.sh (all ~4k frames of seq 08,
# ~300M points/condition), on the cov-shift ep10 extractor by default:
#
#   * naive TTA  : ridge refit on the corrupted pool with frozen-probe
#                  pseudo-labels (NO gate) and the top-50% conf-gated variant
#   * naive AL   : plain ridge on the 56+500 random bank with TRUE labels
#   * memory-bank AL (current method): W_res pseudo/true (oracle U r=8)
#   * ceiling W* / frozen W0 (R4) + proto frozen/ceiling (R1)
#   * per-class nearest-mean separability in the RAW 128-d features AND the
#     binarized 10000-d HDC code (clean vs pool prototypes), plus a held-out
#     clean-reservoir reference
#
# Usage:
#   bash run_al_full_naive.sh 3                # cov-shift ep10, all 8 conds
#   CONDS=fog,crosstalk bash run_al_full_naive.sh 3
#   MAX_FRAMES=200 bash run_al_full_naive.sh 3 # quick smoke test
#   EXTRACTORS="cov_ep21:<method>:<ckpt>" bash run_al_full_naive.sh 3
#
# Output: robust_diagnostic/logs/al_full_naive_ep10.json
#   extractors[<label>].conds[<cond>] = { linear_frozen/ceiling/selftrain/
#     selftrain_conf/randbank/W_res_pseudo/W_res_true (+delta), proto_frozen/
#     ceiling, sep_code_clean/pool, sep_feat_clean/pool, sep_*_clean_ref }

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
MAX_FRAMES="${MAX_FRAMES:-0}"
EXTRACTORS="${EXTRACTORS:-cov_ep10:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan}"
SUFFIX="ep10"
[ "$MAX_FRAMES" != "0" ] && SUFFIX="${SUFFIX}_f${MAX_FRAMES}"
echo "Using GPU $GPU (conds=$CONDS, max_frames=$MAX_FRAMES)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_full_naive_diag.py \
  --label "full_naive_${SUFFIX}" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --extractors "$EXTRACTORS" \
  --out "robust_diagnostic/logs/al_full_naive_${SUFFIX}.json" \
  2>&1 | tee "logs/al_full_naive_${SUFFIX}.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== FULL NAIVE OK ==="
  echo "Check robust_diagnostic/logs/al_full_naive_${SUFFIX}.json:"
  echo "  naive self-train should NOT beat frozen (the Iteration 9-12 closure at scale)"
  echo "  naive random-bank AL should be weak vs W_res (the memory-bank method)"
  echo "  per-class sep_code_clean vs sep_feat_clean: where the minority weakness lives"
else
  echo "=== FULL NAIVE FAILED (exit $RC) ==="
  exit $RC
fi
