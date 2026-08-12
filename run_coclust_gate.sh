#!/usr/bin/env bash
# Gate check for the corrupted-only clustering term (Iteration-12 ceiling design):
# micro-train it at two weights, then verify the MECHANISM in the feature space before
# committing to a 10h medium run.
#
# Safety indicators (printed by the extractor_diff at the end), coclust vs the
# corsupcon micro reference on FOG:
#   - corr_tightness UP    -> intra-corrupted packing improved (ceiling driver #1)
#   - dir_retention NOT ~1 -> the shifted direction is retained (ceiling driver #2)
#   - al_purity UP         -> active-learning readiness improved
#   - car fog oracle UP    -> the ceiling itself moves
# and from the scale_gap autopsies: the crosstalk naive gap is maintained (~>= 0.3)
# while the fog gap does not get worse.
#
# Usage:
#   bash run_coclust_gate.sh            # GPU 3

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_abl() {
  local method="$1"; local logdir="$2"; local label="$3"
  echo "=== [$label] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs 12 --cutoff 0.1 --log_dir "$logdir" \
    2>&1 | tee "logs/dglsspp_${label}_micro.log" || fail "train $label"
  echo "=== [$label] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "$method" --path "$logdir/$method" --label "${label}_micro" \
    2>&1 | tee "logs/dglsspp_${label}_micro_diag.log" || fail "eval $label"
}

# reference: full corsupcon micro (already trained, eval-only)
echo "=== [corsupcon ref] scale_gap autopsy (eval-only) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
  --method "supcon_vib_dglsspp_corsupcon" \
  --path "robust_diagnostic/logs/micro_corsupcon/supcon_vib_dglsspp_corsupcon" \
  --label "corsupcon_micro" \
  2>&1 | tee "logs/dglsspp_corsupcon_micro_diag.log" || fail "ref"

run_abl "supcon_vib_dglsspp_corsupcon_coclust" "robust_diagnostic/logs/micro_coclust" "corsupcon_coclust"
run_abl "supcon_vib_dglsspp_corsupcon_coclust_w005" "robust_diagnostic/logs/micro_coclust_w005" "corsupcon_coclust_w005"

echo "=== feature-space mechanism check: corsupcon (micro) vs coclust (micro) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
  --path_a "robust_diagnostic/logs/micro_corsupcon/supcon_vib_dglsspp_corsupcon" \
  --method_a "supcon_vib_dglsspp_corsupcon" --label_a "corsupcon_micro" \
  --path_b "robust_diagnostic/logs/micro_coclust/supcon_vib_dglsspp_corsupcon_coclust" \
  --method_b "supcon_vib_dglsspp_corsupcon_coclust" --label_b "coclust_micro" \
  --out "robust_diagnostic/logs/extractor_diff_coclust_gate.json" \
  2>&1 | tee "logs/extractor_diff_coclust_gate.log" || fail "gate diff"

echo ""
echo "=== GATE SUMMARY (vs corsupcon micro) ==="
echo "From logs/extractor_diff_coclust_gate.log (FOG): look for corr_tightness UP,"
echo "  dir_retention retained (not ~1.0), al_purity UP, car fog oracle UP."
echo "From logs/dglsspp_corsupcon_coclust_micro_diag.log: crosstalk naive gap ~>= 0.3,"
echo "  fog gap not worse than the reference."
echo "If all hold at BOTH coclust weights, the 10h medium run is a safe bet:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods supcon_vib_dglsspp_corsupcon_coclust --epochs 21 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_coclust"
