#!/usr/bin/env bash
# Combined: (1) train the cov-shift extractor to exactly 10 epochs (the optimal window
# found by the monitor), then (2) run the full performance battery on BOTH the ep-10
# model and the ep-21 model, so we can compare and pick the more robust version.
#
# This gives the complete per-condition picture (full TTA battery, frozen labeled
# ceiling, gate structure, extractor-diff vs DGLSS++/Robust) for both checkpoints,
# which fills the README tables at both the optimal window and the full run.
#
# Timing: ~4h pretraining + ~1.5-2h battery per model = ~7-8h total.
#
# Usage:
#   bash run_covshift_full.sh 3          # GPU 3, train ep-10 then battery both models

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_DIR="robust_diagnostic/logs/ep10_$METHOD"
EP10_CKPT="$EP10_DIR/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"
DGLSSPP_METHOD="supcon_vib_dglsspp"
ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"
ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
echo "Using GPU $GPU"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# ---------- [1] train the ep-10 model (optimal window) ----------
echo "=== [1/2] pretrain cov-shift to exactly 10 epochs / 100% (~4h) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$METHOD" --epochs 10 --cutoff 1.0 \
  --log_dir "$EP10_DIR" \
  2>&1 | tee "logs/covshift_ep10_train.log" || fail "ep10 train"

echo "=== [1b] verify ep-10 stopped at the right epoch ==="
python3 -c "
import torch
w = torch.load('$EP10_CKPT/SENet', map_location='cpu')
print('ep-10 checkpoint dict epoch:', w['epoch'], '(expect 9 = 10 epochs trained)')
" || echo "(epoch read failed, check log manually)"

# ---------- [2] full battery on both models ----------
run_battery() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== battery on $label ($ckpt) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/tta_ceiling_diag.py \
    --path "$ckpt" --method "$METHOD" --label "$label" \
    2>&1 | tee "logs/tta_ceiling_${label}.log" || fail "tta $label"
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/frozen_ceiling_diag.py \
    --path "$ckpt" --method "$METHOD" --label "$label" \
    2>&1 | tee "logs/frozen_ceiling_${label}.log" || fail "frozen $label"
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/gate_structure_diag.py \
    --path "$ckpt" --method "$METHOD" --label "$label" \
    2>&1 | tee "logs/gate_structure_${label}.log" || fail "gate $label"
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
    --path_b "$ckpt" --method_b "$METHOD" --label_b "$label" \
    --out "robust_diagnostic/logs/extractor_diff_${label}_vs_dglsspp.json" \
    2>&1 | tee "logs/extractor_diff_${label}_vs_dglsspp.log" || fail "diffdglsspp $label"
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
    --path_b "$ckpt" --method_b "$METHOD" --label_b "$label" \
    --out "robust_diagnostic/logs/extractor_diff_${label}_vs_robust.json" \
    2>&1 | tee "logs/extractor_diff_${label}_vs_robust.log" || fail "diffrobust $label"
}

echo "=== [2/2] battery on the ep-21 model ==="
run_battery "$EP21_CKPT" "covshift_ep21"

echo "=== [2/2] battery on the ep-10 model ==="
run_battery "$EP10_CKPT" "covshift_ep10"

echo ""
echo "=== FULL COMPARISON SUMMARY ==="
echo "Compare covshift_ep10 vs covshift_ep21 across every log:"
echo "  - tta_ceiling_*.log    : the full TTA levers (naive/conf/dist/BN/kNN) per cond"
echo "  - frozen_ceiling_*.log : the labeled ceiling (hdc_oracle) per condition"
echo "  - gate_structure_*.log : per-signal gate AUROC"
echo "  - extractor_diff_*.log : per-class ceiling + TTA vs DGLSS++ and Robust"
echo "Pick the version (ep10 or ep21) with the higher fog/crosstalk ceiling AND the"
echo "higher TTA across the battery -- that is the paper's extractor. Then the README"
echo "tables use that model's numbers."
