#!/usr/bin/env bash
# Run the full TTA diagnostic suite (the ones previously used on the supcon_vib
# setup) on the Robust DGLSS++ 21ep extractor. Eval-only, ~1h total.
#
#   1. gate_structure_diag : per-signal correct-vs-wrong AUROC (conf/entr/dist/norm/
#                            margin/density/fusion) + confident-wrong recoverability
#   2. ttagate_diag        : the gated prototype update (norm gate, as for DGLSS++)
#   3. tta_ceiling_diag    : the full battery (assignment wall + naive/conf/dist/BN/kNN)
#   4. frozen_ceiling_diag : the labeled ceiling (LP + HDC oracle) per condition
#
# Usage:
#   bash run_robust_tta.sh            # GPU 3

set -u
GPU="${1:-3}"
CKPT="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
METHOD="supcon_vib_dglsspp_corsupcon"
LABEL="robust_21ep"
echo "Using GPU $GPU  (checkpoint: $CKPT)"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

echo "=== [1/4] gate-structure (per-signal AUROC) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/gate_structure_diag.py \
  --path "$CKPT" --method "$METHOD" --label "$LABEL" \
  2>&1 | tee "logs/gate_structure_${LABEL}.log" || fail "gate structure"

echo "=== [2/4] ttagate (norm-gated update) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/ttagate_diag.py \
  --path "$CKPT" --method "$METHOD" --label "$LABEL" \
  2>&1 | tee "logs/ttagate_${LABEL}.log" || fail "ttagate"

echo "=== [3/4] tta-ceiling (full battery) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/tta_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label "$LABEL" \
  2>&1 | tee "logs/tta_ceiling_${LABEL}.log" || fail "tta ceiling"

echo "=== [4/4] frozen-ceiling (labeled ceiling) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/frozen_ceiling_diag.py \
  --path "$CKPT" --method "$METHOD" --label "$LABEL" \
  2>&1 | tee "logs/frozen_ceiling_${LABEL}.log" || fail "frozen ceiling"

echo "All done."
