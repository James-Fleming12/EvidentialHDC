#!/usr/bin/env bash
# Iteration-19.8 diagnostic: the anti-anchor test.
#
# The common-failure diagnosis: every objective we built (GMSIFC/LSCC/SupCon-anchor/
# dircons/corrsc/hdc) ERASES the corruption shift, while plain DGLSS++ keeps its
# ceiling precisely because it was never told to undo corruption. This runs the ONE
# objective none of the failed variants tried -- an ANTI-anchor that penalizes the
# corrupted->clean cosine so the shift is retained.
#
#   supcon_vib_dglsspp_antianchor : plain DGLSS++ (beam-drop view, GMSIFC+LSCC, no
#                                   SupCon) + 0.1 * (corrupted->clean cosine penalty)
#
# Because micro is unreliable (three documented micro-to-medium reversals), this runs
# at MEDIUM-LITE (12 ep / 100%) so the direction is actually testable, then gates on
# the per-class shift-retention structure vs plain DGLSS++:
#   PASS: car fog dir_retention < 0.5 (the shift is retained, like plain 0.37) AND
#         fog/crosstalk oracle not below plain DGLSS++.
#   If dir_retention stays ~0.8-0.9, the erasure is baked into GMSIFC/LSCC themselves
#   (not just the anchor), and the "never told to undo corruption" hypothesis is wrong
#   -- the next step is a no-consistency DGLSS++ (CE-only) baseline.
#
# Usage:
#   bash run_antianchor.sh 3           # GPU 3, 12 ep medium-lite (~6h)
#   bash run_antianchor.sh 3 21        # full medium (~10h)

set -u
GPU="${1:-3}"
EPOCHS="${2:-12}"
METHOD="supcon_vib_dglsspp_antianchor"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"   # plain DGLSS++ medium
DGLSSPP_METHOD="supcon_vib_dglsspp"
echo "Using GPU $GPU, $EPOCHS ep / 100% (medium-lite)"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/2] train anti-anchor ($EPOCHS ep / 100%) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$METHOD" --epochs "$EPOCHS" --cutoff 1.0 \
  --log_dir "robust_diagnostic/logs/med_$METHOD" \
  2>&1 | tee "logs/antianchor_med_train.log" || fail "train"

echo "=== [2/2] extractor_diff: anti-anchor vs plain DGLSS++ (per-class structure) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
  --path_b "robust_diagnostic/logs/med_$METHOD/$METHOD" --method_b "$METHOD" --label_b "antianchor_med" \
  --out "robust_diagnostic/logs/extractor_diff_antianchor.json" \
  2>&1 | tee "logs/antianchor_extractordiff.log" || fail "gate"

echo ""
echo "=== ANTI-ANCHOR VERDICT ==="
echo "From logs/antianchor_extractordiff.log (fog), check car(4):"
echo "  - dir_retention < 0.5 -> the shift is retained (like plain DGLSS++ 0.37)"
echo "  - oracle NOT below plain DGLSS++ (car fog 0.303) -> the erasure was the ceiling killer"
echo "If dir_retention stays ~0.8-0.9, the erasure is in GMSIFC/LSCC themselves;"
echo "next step = a CE-only (no-consistency) DGLSS++ baseline to isolate which term erases it."
