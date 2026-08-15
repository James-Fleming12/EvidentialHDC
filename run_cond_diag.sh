#!/usr/bin/env bash
# Run the healthy-condition ceiling diagnostic on the cov-shift model(s).
# For each checkpoint, compares per-class feature structure (feat_cos / dir_retention
# / corr_tightness / zs) on snow + wet_ground vs plain DGLSS++, to find which classes
# lose recoverable structure and whether it is direction or packing.
#
# Usage:
#   bash run_cond_diag.sh 3                            # GPU 3, ep10 + ep21 cov-shift
#   bash run_cond_diag.sh 3 ep10                       # only the ep10 model

set -u
GPU="${1:-3}"
ONLY="${2:-all}"
echo "Using GPU $GPU"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_cond() {
  local ckpt="$1"; local label="$2"
  echo "=== cond_structure diag on $label (snow + wet_ground) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label_b "$label" \
    --conds snow,wet_ground \
    --out "robust_diagnostic/logs/cond_structure_$label.json" \
    2>&1 | tee "logs/cond_structure_$label.log" || fail "cond $label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_cond "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_cond "$EP21_CKPT" "covshift_ep21"
fi

echo ""
echo "=== VERDICT ==="
echo "From logs/cond_structure_*.log, look for classes where:"
echo "  - dir_ret or corr_tight drops under cov-shift (B) -> the normalization erased"
echo "    that class's recoverable structure"
echo "  - zs (frozen decode) drops too -> the loss is real, not a re-estimation artifact"
echo "If the loss is on dir_ret (direction): preserve anisotropy before InstanceNorm."
echo "If on corr_tight (packing): scale features to clean scale first."
