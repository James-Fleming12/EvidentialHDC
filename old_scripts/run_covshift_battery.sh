#!/usr/bin/env bash
# Full performance battery for the cov-shift DGLSS++ extractor (ep-21 checkpoint).
# Gives the complete per-condition picture: full TTA battery, frozen labeled ceiling,
# gate structure, and the extractor-diff comparison vs DGLSS++ / Robust.
#
# Note: the README tables currently use the ep-10 model (the optimal window) and only
# the naive-EMA lever. This battery runs on the ep-21 checkpoint for now; rerun on the
# ep-10 checkpoint (ep10_supcon_vib_dglsspp_inputin_in_chan) once the 4h rerun lands
# to fill the tables at the optimal window.
#
# Usage:
#   bash run_covshift_battery.sh 3          # GPU 3

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"
DGLSSPP_METHOD="supcon_vib_dglsspp"
ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
echo "Using GPU $GPU  (checkpoint: $CKPT)"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/5] full TTA battery (naive/conf/dist/BN/kNN) per condition ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/tta_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label covshift_med \
  2>&1 | tee "logs/tta_ceiling_covshift_med.log" || fail "tta ceiling"

echo "=== [2/5] frozen labeled ceiling per condition ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/frozen_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label covshift_med \
  2>&1 | tee "logs/frozen_ceiling_covshift_med.log" || fail "frozen ceiling"

echo "=== [3/5] gate-signal structure per condition ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/gate_structure_diag.py \
  --path "$CKPT" --method "$METHOD" --label covshift_med \
  2>&1 | tee "logs/gate_structure_covshift_med.log" || fail "gate structure"

echo "=== [4/5] extractor_diff vs plain DGLSS++ (per-class ceiling + TTA) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
  --path_b "$CKPT" --method_b "$METHOD" --label_b "covshift_med" \
  --out "robust_diagnostic/logs/extractor_diff_covshift_vs_dglsspp.json" \
  2>&1 | tee "logs/extractor_diff_covshift_vs_dglsspp.log" || fail "diff vs dglsspp"

echo "=== [5/5] extractor_diff vs Robust DGLSS++ (per-class ceiling + TTA) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
  --path_b "$CKPT" --method_b "$METHOD" --label_b "covshift_med" \
  --out "robust_diagnostic/logs/extractor_diff_covshift_vs_robust.json" \
  2>&1 | tee "logs/extractor_diff_covshift_vs_robust.log" || fail "diff vs robust"

echo ""
echo "=== BATTERY SUMMARY ==="
echo "From logs/tta_ceiling_covshift_med.log: the full TTA levers per condition."
echo "From logs/frozen_ceiling_covshift_med.log: the labeled ceiling (hdc_oracle) per"
echo "  condition -- this fills the -- blanks in the README zero-shot table."
echo "From logs/gate_structure_covshift_med.log: the per-signal AUROC (gate quality)."
echo "From logs/extractor_diff_covshift_vs_*.log: per-class ceiling + TTA vs the baselines."
