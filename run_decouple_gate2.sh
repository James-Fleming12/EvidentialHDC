#!/usr/bin/env bash
# Decoupling micro-gate 2 (Iteration-16 follow-up): the first two-branch gate failed
# because the corr branch never retained a shifted direction -- the loss routing gave
# it no freedom (LSCC is a clean-view alignment term, and CE on the concat re-pulls it).
# This gate tests the two fixes, still BEFORE any 10h medium commitment:
#
#   Q1 (is the corr branch fixable by removing its anchors?):
#       twobranch_128_64_corrfree -- LSCC DROPPED on the corr slice; CE on the full
#       concat is the corr branch's only clean pull (genuinely free capacity).
#   Q2 (does an explicit displacement-direction objective retain the shift?):
#       residual_128_128_dircons -- residual form + L_dir = 1 - cos(dz, sg(delta_c))
#       with a per-class EMA displacement direction (idea #3: same-class corrupted
#       points move coherently; direction-only, the weakest structural commitment).
#
# Two micro variants (12 ep / 10% data, ~1-1.5h each) vs the corsupcon micro reference.
# The extractor_diff GATE prints per-branch structure with --inv_ch 128:
#   PASS signals (vs corsupcon reference, on FOG):
#     - corr_dir_retention < 1 (the corr branch keeps the shifted direction)
#     - inv_feat_cos stays high (the anchor still lands on the invariant branch)
#     - car fog oracle UP (the concatenated ceiling moves)
#     - corr_tightness not collapsed
#   and from the scale_gap autopsies: the crosstalk naive gap is maintained
#   (~>= 0.3) while the fog gap does not get worse.
#
# Usage:
#   bash run_decouple_gate2.sh            # GPU 3

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
VARIANTS="supcon_vib_dglsspp_corsupcon_twobranch_128_64_corrfree supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# micro-train + scale_gap autopsy for one variant
run_abl() {
  local method="$1"; local logdir="$2"; local label="$3"
  echo "=== [$label] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs 12 --cutoff 0.1 --log_dir "$logdir" \
    2>&1 | tee "logs/decouple2_${label}_micro.log" || fail "train $label"
  echo "=== [$label] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "$method" --path "$logdir/$method" --label "${label}_micro" \
    2>&1 | tee "logs/decouple2_${label}_micro_diag.log" || fail "eval $label"
}

for v in $VARIANTS; do
  run_abl "$v" "robust_diagnostic/logs/micro_$v" "$v"
done

# mechanism gate: each variant vs the reference, per-branch structure
for v in $VARIANTS; do
  echo "=== [gate] extractor_diff $v vs corsupcon micro (--inv_ch 128) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "robust_diagnostic/logs/micro_$v/$v" --method_b "$v" --label_b "$v" \
    --inv_ch 128 --out "robust_diagnostic/logs/decouple2_gate_$v.json" \
    2>&1 | tee "logs/decouple2_gate_$v.log" || fail "gate $v"
done

echo ""
echo "=== GATE SUMMARY (vs corsupcon micro) ==="
echo "From logs/decouple2_gate_*.log (FOG), look for PASS on each variant:"
echo "  - corr_dir_retention < 1       (corr branch keeps the shifted direction)"
echo "  - inv_feat_cos still ~high     (invariant branch stays anchored)"
echo "  - car fog oracle UP            (concatenated ceiling moved)"
echo "  - corr_tightness not collapsed"
echo "From logs/decouple2_*_micro_diag.log: crosstalk naive gap ~>= 0.3, fog gap not worse."
echo "If a variant holds all signals, its 10h medium run is a safe bet:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods <method> --epochs 21 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_<method>"
