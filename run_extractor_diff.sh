#!/usr/bin/env bash
# Run the extractor-diff feature-space comparisons (plain DGLSS++ / robust DGLSS++ /
# soft-anchor blend05) on one GPU, tee-ing each to its own log. Eval-only.
#
# Run 1: robust DGLSS++ (21ep) vs the soft-anchor blend05 medium -> the per-class
#        label-ceiling drivers (blend05 has the highest crosstalk / lowest fog oracle).
# Run 2: plain DGLSS++ vs robust DGLSS++ -> the anchoring effect + active-learning
#        readiness (one-label-per-cluster purity) for the AL framework.
#
# Usage:
#   bash run_extractor_diff.sh            # GPU 3
#   bash run_extractor_diff.sh 0          # GPU 0

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/2] robust DGLSS++ vs blend05 (label-ceiling drivers) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon" \
  --method_a "supcon_vib_dglsspp_corsupcon" --label_a "corsupcon_med" \
  --path_b "robust_diagnostic/logs/med_blend05/supcon_vib_dglsspp_corsupcon_blend05" \
  --method_b "supcon_vib_dglsspp_corsupcon_blend05" --label_b "blend05_med" \
  --out "robust_diagnostic/logs/extractor_diff_corsupcon_blend05.json" \
  2>&1 | tee "logs/extractor_diff_corsupcon_blend05.log" || fail "run 1"

echo "=== [2/2] plain DGLSS++ vs robust DGLSS++ (anchoring + AL readiness) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "robust_diagnostic/logs/supcon_vib_dglsspp" \
  --method_a "supcon_vib_dglsspp" --label_a "dglsspp_med" \
  --path_b "robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon" \
  --method_b "supcon_vib_dglsspp_corsupcon" --label_b "corsupcon_med" \
  --out "robust_diagnostic/logs/extractor_diff_dglsspp_corsupcon.json" \
  2>&1 | tee "logs/extractor_diff_dglsspp_corsupcon.log" || fail "run 2"

echo "All done."
