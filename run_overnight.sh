#!/usr/bin/env bash
# Overnight broad sweep (<14h) of ceiling-raising directions (Iteration 19.5 +
# feedback). Each direction is micro-trained in its OWN isotropy call so a crash in
# one does not kill the batch, then gated with extractor_diff --inv_ch 128.
#
# Directions tested (feedback order):
#   1. dircons closure   : L_res 0.01 / 0.02 (the residual-penalty lever, last micro)
#   2. corrsc            : corruption-manifold multi-positive SupCon (same class ->
#                          same corrupted manifold; supcon(aug2, aug), weak clean term)
#   3. corrfree_corrsc   : corrfree base + corrupted-manifold supervision on the corr
#                          slice (freedom + structure, the corrfree failure fix)
#   4. hdc               : HDC-aware soft-prototype loss (margin in the binarized
#                          geometry, the exact space the decoder reads)
#   5. concat_diag (eval): the teacher-premise falsification -- do the frozen robust +
#                          plain DGLSS++ features combine in one HDC decoder at all?
#
# The decision rule (read after, judgment not threshold):
#   - oracle UP (car corr_dir < 1 for the dircons levers; any ceiling gain for the
#     others) WITHOUT naive TTA collapsing AND healthy conditions not down.
#   - If NO direction moves the ceiling -> representation line closed, AL framework
#     on the robust extractor, teacher run not warranted.
#   - If a direction is clearly positive -> promote to a medium run (separate script).
#
# Usage:
#   bash run_overnight.sh 3                # GPU 3, all micro (8 ep / 10%)
#   bash run_overnight.sh 3 12             # custom epochs

set -u
GPU="${1:-3}"
EPOCHS="${2:-8}"
echo "Using GPU $GPU, $EPOCHS ep / 10% per variant"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
ROBUST_PATH="robust_diagnostic/logs/med_corsupcon_21ep/$REF_METHOD"
ROBUST_METHOD="supcon_vib_dglsspp_corsupcon"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"
DGLSSPP_METHOD="supcon_vib_dglsspp"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# ---------- 1. dircons closure ----------
DIRCONS_RES="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_res01"
DIRCONS_RES2="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_res02"
echo "=== [1] dircons L_res sweep (res01, res02) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$DIRCONS_RES,$DIRCONS_RES2" --epochs "$EPOCHS" --cutoff 0.1 \
  --log_dir "robust_diagnostic/logs/micro_dircons_res" \
  2>&1 | tee "logs/overnight_dircons_res_micro.log" || fail "dircons res"
for v in dircons_res01 dircons_res02; do
  m="supcon_vib_dglsspp_corsupcon_residual_128_128_$v"
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "robust_diagnostic/logs/micro_dircons_res/$m" --method_b "$m" --label_b "$v" \
    --frames 50 --pool_size 50000 --val_size 50000 --inv_ch 128 \
    --out "robust_diagnostic/logs/overnight_gate_$v.json" \
    2>&1 | tee "logs/overnight_gate_$v.log" || fail "gate $v"
done

# ---------- 2. corrsc ----------
CORRSC="supcon_vib_dglsspp_corsupcon_corrsc"
echo "=== [2] corrsc: corruption-manifold multi-positive SupCon ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$CORRSC" --epochs "$EPOCHS" --cutoff 0.1 \
  --log_dir "robust_diagnostic/logs/micro_corrsc" \
  2>&1 | tee "logs/overnight_corrsc_micro.log" || fail "corrsc train"
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
  --path_b "robust_diagnostic/logs/micro_corrsc/$CORRSC" --method_b "$CORRSC" --label_b "corrsc" \
  --frames 50 --pool_size 50000 --val_size 50000 \
  --out "robust_diagnostic/logs/overnight_gate_corrsc.json" \
  2>&1 | tee "logs/overnight_gate_corrsc.log" || fail "corrsc gate"

# ---------- 3. corrfree_corrsc ----------
CORRFREE_CORRSC="supcon_vib_dglsspp_corsupcon_corrfree_corrsc"
echo "=== [3] corrfree_corrsc: free corr head + corrupted-manifold supervision ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$CORRFREE_CORRSC" --epochs "$EPOCHS" --cutoff 0.1 \
  --log_dir "robust_diagnostic/logs/micro_corrfree_corrsc" \
  2>&1 | tee "logs/overnight_corrfree_corrsc_micro.log" || fail "corrfree_corrsc train"
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
  --path_b "robust_diagnostic/logs/micro_corrfree_corrsc/$CORRFREE_CORRSC" \
  --method_b "$CORRFREE_CORRSC" --label_b "corrfree_corrsc" \
  --frames 50 --pool_size 50000 --val_size 50000 --inv_ch 128 \
  --out "robust_diagnostic/logs/overnight_gate_corrfree_corrsc.json" \
  2>&1 | tee "logs/overnight_gate_corrfree_corrsc.log" || fail "corrfree_corrsc gate"

# ---------- 4. hdc ----------
HDC="supcon_vib_dglsspp_corsupcon_hdc"
echo "=== [4] hdc: HDC-aware soft-prototype loss ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$HDC" --epochs "$EPOCHS" --cutoff 0.1 \
  --log_dir "robust_diagnostic/logs/micro_hdc" \
  2>&1 | tee "logs/overnight_hdc_micro.log" || fail "hdc train"
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
  --path_b "robust_diagnostic/logs/micro_hdc/$HDC" --method_b "$HDC" --label_b "hdc" \
  --frames 50 --pool_size 50000 --val_size 50000 \
  --out "robust_diagnostic/logs/overnight_gate_hdc.json" \
  2>&1 | tee "logs/overnight_gate_hdc.log" || fail "hdc gate"

# ---------- 5. teacher-premise falsification (eval-only) ----------
echo "=== [5] concat_diag: do frozen robust + plain DGLSS++ combine in one decoder? ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/concat_diag.py \
  --path_a "$ROBUST_PATH" --method_a "$ROBUST_METHOD" --label_a "robust_21ep" \
  --path_b "$DGLSSPP_PATH" --method_b "$DGLSSPP_METHOD" --label_b "dglsspp_med" \
  --inv_ch 128 --out "robust_diagnostic/logs/concat_diag_results.json" \
  2>&1 | tee "logs/overnight_concat_diag.log" || fail "concat diag"

echo ""
echo "=== OVERNIGHT SWEEP SUMMARY ==="
echo "Compare each gate's extractor_diff vs the corsupcon reference:"
echo "  - dircons_res01/res02 : car corr_dir < 1 (residual finally develops)?"
echo "  - corrsc / corrfree_corrsc / hdc : oracle UP on fog/crosstalk WITHOUT naive"
echo "    TTA collapsing and healthy conditions down?"
echo "  - concat_diag : concat oracle > max(A,B) oracle AND concat naive >= A naive?"
echo "Decision: no direction moves the ceiling -> representation line closed, use the"
echo "AL framework on the robust extractor. A clearly-positive direction -> promote it"
echo "to a medium run (separate script, judgment not a threshold)."
