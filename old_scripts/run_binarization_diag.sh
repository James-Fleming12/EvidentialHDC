#!/usr/bin/env bash
# Binarization diagnostic on the cov-shift extractor(s): quantify what the current
# sign-binarization loses on the healthy conditions (pre-sign margin fraction) and
# test three alternative encodings (per-coordinate bias, z-score, fourier features)
# on the SAME frozen features. Run on both the ep-10 and ep-21 cov-shift models vs
# plain DGLSS++.
#
# Usage:
#   bash run_binarization_diag.sh 3              # GPU 3, ep10 + ep21
#   bash run_binarization_diag.sh 3 ep10         # only ep10

set -u
GPU="${1:-3}"
ONLY="${2:-all}"
echo "Using GPU $GPU"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_bin() {
  local ckpt="$1"; local label="$2"
  echo "=== binarization diag on $label ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/binarization_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label_b "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/binarization_$label.json" \
    2>&1 | tee "logs/binarization_$label.log" || fail "bin $label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_bin "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_bin "$EP21_CKPT" "covshift_ep21"
fi

echo ""
echo "=== VERDICT ==="
echo "A. margin_frac (clean pre-sign near-0 fraction): if B > A on snow/wet_ground,"
echo "   the cov-shift healthy features sit closer to the sign threshold (packing loss)."
echo "B. On the healthy conditions, does 'bias' / 'zscore' / 'fourier' recover the"
echo "   B oracle toward A WITHOUT losing the fog/crosstalk B gain? That encoding is"
echo "   the fix for the healthy-ceiling regression."
