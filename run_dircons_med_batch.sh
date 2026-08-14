#!/usr/bin/env bash
# Medium-scale comparison batch for the dircons ceiling problem (Iteration 19.5).
#
# The w02 derisk micro FAILED: stronger pull (0.2) does not reach car (corr_dir
# ~0.95) and the oracle falls. So instead of a single 10h bet, compare the three
# remaining levers AT MEDIUM SCALE, in parallel (one GPU each), each at a reduced
# epoch count so they all finish overnight:
#
#   dircons_w02_res01 : L_res 0.05 -> 0.01 (let car's residual actually develop)
#   dircons_frag_w02  : fragile-only dircons (2/7/13/14/15) at dir_w 0.2 (concentrate
#                       the direction on the ceiling-relevant classes)
#   dircons_frag      : fragile-only at dir_w 0.1 (the healthy-condition decoupling)
#
# Each is evaluated with the same battery: extractor_diff (--inv_ch 128) vs the
# robust 21ep baseline, + tta_ceiling + frozen_ceiling, so the batch directly
# answers which lever (if any) raises the crosstalk ceiling past DGLSS++ (0.214)
# while keeping TTA and the healthy conditions.
#
# Usage:
#   bash run_dircons_med_batch.sh                # GPUs 3 4 5, 17 epochs each
#   bash run_dircons_med_batch.sh "3 4" 17       # 2 GPUs, custom epochs
#   bash run_dircons_med_batch.sh "3" 21         # 1 GPU, sequential full medium

set -u
GPUS="${1:-3 4 5}"
EPOCHS="${2:-17}"
METHODS="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_w02_res01 supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_frag_w02 supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_frag"

read -r -a GPU_LIST <<< "$GPUS"
read -r -a M_LIST <<< "$METHODS"
N="${#GPU_LIST[@]}"
echo "Using GPUs: $GPUS  ($N workers, $EPOCHS epochs each)"

if [ "${#M_LIST[@]}" -ne "$N" ]; then
  echo "ERROR: need one GPU per method ($N GPUs, ${#M_LIST[@]} methods)" >&2
  exit 1
fi

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }
PIDS=()

for i in $(seq 0 $((N - 1))); do
  GPU="${GPU_LIST[$i]}"
  METHOD="${M_LIST[$i]}"
  MED_DIR="robust_diagnostic/logs/med_$METHOD"
  echo "=== [$i] GPU $GPU: training $METHOD ($EPOCHS ep / 100%) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$METHOD" --epochs "$EPOCHS" --cutoff 1.0 --log_dir "$MED_DIR" \
    2>&1 | tee "logs/medbatch_${METHOD}_train.log" &
  PIDS+=($!)
done

echo "=== waiting for all trainings (${#PIDS[@]} workers) ==="
for pid in "${PIDS[@]}"; do wait "$pid" || fail "a training worker"; done
echo "=== all trainings done ==="

for METHOD in "${M_LIST[@]}"; do
  CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
  ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
  ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
  echo "=== battery: $METHOD ==="
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
    --path_b "$CKPT" --method_b "$METHOD" --label_b "$METHOD" \
    --inv_ch 128 --out "robust_diagnostic/logs/extractor_diff_medbatch_$METHOD.json" \
    2>&1 | tee "logs/medbatch_${METHOD}_extractordiff.log" || fail "extractor_diff $METHOD"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} uv run python robust_diagnostic/tta_ceiling_diag.py \
    --path "$CKPT" --method "$METHOD" --label "$METHOD" \
    2>&1 | tee "logs/medbatch_${METHOD}_ttaceiling.log" || fail "tta_ceiling $METHOD"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} uv run python robust_diagnostic/frozen_ceiling_diag.py \
    --path "$CKPT" --method "$METHOD" --label "$METHOD" \
    2>&1 | tee "logs/medbatch_${METHOD}_frozen.log" || fail "frozen $METHOD"
done

echo ""
echo "=== BATCH SUMMARY ==="
echo "Compare each variant's extractor_diff vs robust_21ep:"
echo "  - crosstalk oracle vs DGLSS++ 0.214 (does ANY lever pass it?)"
echo "  - car corr_dir < 1 (did the residual develop?)"
echo "  - crosstalk naive/BN TTA maintained"
echo "  - healthy-condition ceiling not down (frozen_ceiling snow/wet_ground)"
echo "If a variant passes, commit the 10h full run:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods <method> --epochs 21 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_<method>"
