#!/usr/bin/env bash
# Decoupling micro-gate (Iteration-15 shortlist): validate the two-branch bottleneck
# BEFORE any 10h medium commitment. Two questions:
#
#   Q1 (is decoupling useful at all?): does giving the corruption its OWN capacity
#      (not pulled by the SupCon clean-anchor) retain the shifted, recoverable class
#      structure -- raising the labeled ceiling -- while the invariant branch keeps
#      the assignment/TTA?
#   Q2 (best implementation?): independent corr head vs full-capacity corr head vs
#      the residual form z_corr = z_inv + dz.
#
# Three micro variants (12 ep / 10% data, ~1-1.5h each), each vs the corsupcon micro
# reference:
#   twobranch_128_64   inv 128 + corr 64 (ind), total 192   <- Q1: decoupling useful at all?
#   residual_128_128   inv 128 + corr = inv+dz (res), total 256  <- Q2: residual the better impl?
# (twobranch_128_128 dropped from the gate -- corr capacity is a secondary knob; the
#  two kept variants answer both questions, and the gate stays ~3-3.5h total.)
#
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
#   bash run_decouple_gate.sh            # GPU 3

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
VARIANTS="supcon_vib_dglsspp_corsupcon_twobranch_128_64 supcon_vib_dglsspp_corsupcon_residual_128_128"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# micro-train + scale_gap autopsy for one variant
run_abl() {
  local method="$1"; local logdir="$2"; local label="$3"
  echo "=== [$label] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs 12 --cutoff 0.1 --log_dir "$logdir" \
    2>&1 | tee "logs/decouple_${label}_micro.log" || fail "train $label"
  echo "=== [$label] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "$method" --path "$logdir/$method" --label "${label}_micro" \
    2>&1 | tee "logs/decouple_${label}_micro_diag.log" || fail "eval $label"
}

# reference extractor_diff (eval-only, no branch split; 128D)
# NOTE: no ref-vs-ref baseline run -- the corsupcon micro reference is embedded as
# path_a in every gate comparison below.

for v in $VARIANTS; do
  run_abl "$v" "robust_diagnostic/logs/micro_$v" "$v"
done

# mechanism gate: each variant vs the reference, per-branch structure
for v in $VARIANTS; do
  echo "=== [gate] extractor_diff $v vs corsupcon micro (--inv_ch 128) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "robust_diagnostic/logs/micro_$v/$v" --method_b "$v" --label_b "$v" \
    --inv_ch 128 --out "robust_diagnostic/logs/decouple_gate_$v.json" \
    2>&1 | tee "logs/decouple_gate_$v.log" || fail "gate $v"
done

echo ""
echo "=== GATE SUMMARY (vs corsupcon micro) ==="
echo "From logs/decouple_gate_*.log (FOG), look for PASS on each variant:"
echo "  - corr_dir_retention < 1       (corr branch keeps the shifted direction)"
echo "  - inv_feat_cos still ~high     (invariant branch stays anchored)"
echo "  - car fog oracle UP            (concatenated ceiling moved)"
echo "  - corr_tightness not collapsed"
echo "From logs/decouple_*_micro_diag.log: crosstalk naive gap ~>= 0.3, fog gap not worse."
echo "If a variant holds all signals, its 10h medium run is a safe bet:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods <method> --epochs 21 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_<method>"
