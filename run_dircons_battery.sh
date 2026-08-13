#!/usr/bin/env bash
# Post-run full battery for the 10h dircons medium run. Run AFTER
# run_dircons_medium.sh completes. Covers the same suite as the robust_21ep
# run-through plus the extractor_diff per-branch gate, so the medium dircons can be
# compared to the robust 21ep baseline AND the dircons micro gate.
#
# Usage:
#   bash run_dircons_battery.sh          # GPU 3

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"
MED_DIR="robust_diagnostic/logs/med_dircons"
CKPT="$MED_DIR/$METHOD"
ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
echo "Using GPU $GPU  (checkpoint: $CKPT)"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/5] gate-signal structure ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/gate_structure_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/gate_structure_dircons_med.log" || fail "gate structure"

echo "=== [2/5] norm-gated prototype update (TTA) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/ttagate_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/ttagate_dircons_med.log" || fail "ttagate"

echo "=== [3/5] full TTA battery ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/tta_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/tta_ceiling_dircons_med.log" || fail "tta ceiling"

echo "=== [4/5] frozen labeled ceiling ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/frozen_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label dircons_med \
  2>&1 | tee "logs/frozen_ceiling_dircons_med.log" || fail "frozen ceiling"

echo "=== [5/5] extractor_diff: dircons med vs robust 21ep (--inv_ch 128) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
  --path_b "$CKPT" --method_b "$METHOD" --label_b "dircons_med" \
  --inv_ch 128 --out "robust_diagnostic/logs/extractor_diff_dircons_med.json" \
  2>&1 | tee "logs/extractor_diff_dircons_med.log" || fail "extractor diff"

echo "All done."
