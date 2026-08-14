#!/usr/bin/env bash
# Iteration-19.8 micro sweep: the "don't-erase" + covariate-shift candidates.
# Fast micro gates (8 ep / 10%) so we iterate quickly; any variant that clearly
# separates from the corsupcon reference gets promoted to medium-lite. Each is its
# own call so a crash in one does not kill the batch.
#
#   antianchor        : plain DGLSS++ + penalty on corrupted->clean cosine (the
#                       inverse of every objective we tried -- tests whether the
#                       ceiling killer is the erasure itself)
#   instancenorm      : plain DGLSS++ with InstanceNorm (BN running-stats are a
#                       covariate-shift failure; BN-alignment was the best TTA lever)
#   cor_instancenorm  : the robust corruption-view base with InstanceNorm (isolates
#                       the norm effect from the view effect)
#
# Gate: extractor_diff vs the corsupcon micro reference + vs plain DGLSS++ micro,
# checking per-class dir_retention (shift retained?) and oracle.
#
# Usage:
#   bash run_micro_robustness.sh 3            # GPU 3, 8 ep / 10%

set -u
GPU="${1:-3}"
EPOCHS="${2:-8}"
echo "Using GPU $GPU, $EPOCHS ep / 10%"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_one() {
  local method="$1"; local label="$2"
  echo "=== [$label] micro training ($EPOCHS ep / 10%) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff 0.1 \
    --log_dir "robust_diagnostic/logs/micro_$label" \
    2>&1 | tee "logs/micro_${label}_train.log" || fail "train $label"
  echo "=== [$label] extractor_diff vs corsupcon micro ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "robust_diagnostic/logs/micro_$label/$method" --method_b "$method" --label_b "$label" \
    --frames 50 --pool_size 50000 --val_size 50000 \
    --out "robust_diagnostic/logs/micro_gate_$label.json" \
    2>&1 | tee "logs/micro_gate_$label.log" || fail "gate $label"
}

run_one "supcon_vib_dglsspp_antianchor" "antianchor"
run_one "supcon_vib_dglsspp_instancenorm" "instancenorm"
run_one "supcon_vib_dglsspp_cor_instancenorm" "cor_instancenorm"

echo ""
echo "=== MICRO ROBUSTNESS VERDICT ==="
echo "From logs/micro_gate_*.log, check vs corsupcon micro:"
echo "  antianchor       : car fog dir_retention < 0.5 (shift retained) and oracle up?"
echo "  instancenorm     : oracle + naive both stable/up (BN-stats sensitivity removed)?"
echo "  cor_instancenorm : same, on the corruption-view base?"
echo "Any variant clearly above the others -> promote to medium-lite:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods <method> --epochs 12 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_<method>"
