#!/usr/bin/env bash
# dircons medium follow-up (Iteration 18):
#   1. Re-run the three harness diagnostics that CRASHED on the 256D features
#      (stale dim_in=128 bug) -- now fixed: tta_ceiling, frozen_ceiling, ttagate.
#      This also answers "is there another TTA mechanism that works as well?" via
#      the full battery (naive / conf / dist / BN / kNN are all in tta_ceiling).
#   2. Re-run extractor_diff (the pulled JSON was NUL-corrupted) to get a clean
#      per-branch JSON.
#   3. Run the TTA-collapse diagnosis (ttacollapse_diag) on that fresh JSON: does
#      the retained corr shift correlate with the naive-update failure per class?
#
# Run AFTER run_dircons_medium.sh. No retraining -- all steps eval the existing
# checkpoint at robust_diagnostic/logs/med_dircons/<method>.
#
# Usage:
#   bash run_dircons_followup.sh        # GPU 3

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"
MED_DIR="robust_diagnostic/logs/med_dircons"
CKPT="$MED_DIR/$METHOD"
ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
echo "Using GPU $GPU  (checkpoint: $CKPT)"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/4] full TTA battery (naive/conf/dist/BN/kNN) on dircons ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/tta_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/tta_ceiling_dircons_med.log" || fail "tta ceiling"

echo "=== [2/4] frozen labeled ceiling on dircons ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/frozen_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/frozen_ceiling_dircons_med.log" || fail "frozen ceiling"

echo "=== [3/4] norm-gated prototype update on dircons ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/ttagate_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/ttagate_dircons_med.log" || fail "ttagate"

echo "=== [4/4] fresh extractor_diff (clean JSON) + TTA-collapse diagnosis ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
  --path_b "$CKPT" --method_b "$METHOD" --label_b "dircons_med" \
  --inv_ch 128 --out "robust_diagnostic/logs/extractor_diff_dircons_med.json" \
  2>&1 | tee "logs/extractor_diff_dircons_med.log" || fail "extractor diff"

CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/ttacollapse_diag.py \
  --json "robust_diagnostic/logs/extractor_diff_dircons_med.json" \
  --label_a "robust_21ep" --label_b "$METHOD" \
  --out "robust_diagnostic/logs/ttacollapse_dircons_med.json" \
  2>&1 | tee "logs/ttacollapse_dircons_med.log" || fail "collapse diagnosis"

echo "All done."
