#!/usr/bin/env bash
# Run the three DGLSS++ robustness-variant micro diagnostics (SupCon / class-balance /
# VIB) plus the plain-DGLSS base per-class check on a single GPU, tee-ing each step
# to its own log.
#
# Usage:
#   bash run_variant_micro.sh            # GPU 3, everything
#   bash run_variant_micro.sh 0          # GPU 0
#   bash run_variant_micro.sh 3 abl      # GPU 3, ablations + anchoring + dglss check
#   bash run_variant_micro.sh 3 anchor   # GPU 3, ONLY the anchoring-direction tests + dglss check
#
# Each training also runs the full isotropy eval (prints + saves isotropy_results.json
# into the variant's log_dir). After training, the scale_gap per-class autopsy runs for
# the same checkpoint. Logs go to logs/dglsspp_<variant>_micro.log (train + isotropy)
# and logs/dglsspp_<variant>_micro_diag.log (per-class autopsy).

set -u

GPU="${1:-3}"
MODE="${2:-all}"
echo "Using GPU $GPU (mode: $MODE)"

fail() {
  echo "ERROR: $1 failed (exit $?)" >&2
}

run_train() {
  local name="$1"
  echo "=== [$name] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "supcon_vib_dglsspp_$name" --epochs 12 --cutoff 0.1 \
    --log_dir "robust_diagnostic/logs/micro_$name" \
    2>&1 | tee "logs/dglsspp_${name}_micro.log" || fail "train $name"
}

run_eval() {
  local name="$1"
  echo "=== [$name] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "supcon_vib_dglsspp_$name" \
    --path "robust_diagnostic/logs/micro_$name/supcon_vib_dglsspp_$name" \
    --label "${name}_micro" \
    2>&1 | tee "logs/dglsspp_${name}_micro_diag.log" || fail "eval $name"
}

if [ "$MODE" = "all" ]; then
  for name in supcon bal vib; do
    run_train "$name"
    run_eval "$name"
  done
fi

# --- Component ablations of the combined robust variant (micro) ---
# First the full combined variant at micro (the reference), then drop GMSIFC / LSCC /
# both, so "does the DGLSS++ stack earn its place" has a same-scale baseline.
run_abl() {
  local method="$1"
  local logdir="$2"
  local label="$3"
  echo "=== [$label] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs 12 --cutoff 0.1 --log_dir "$logdir" \
    2>&1 | tee "logs/dglsspp_${label}_micro.log" || fail "train $label"
  echo "=== [$label] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "$method" --path "$logdir/$method" --label "${label}_micro" \
    2>&1 | tee "logs/dglsspp_${label}_micro_diag.log" || fail "eval $label"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "abl" ]; then
  run_abl "supcon_vib_dglsspp_corsupcon" "robust_diagnostic/logs/micro_corsupcon" "corsupcon"
  run_abl "supcon_vib_dglsspp_corsupcon_nogmsifc" "robust_diagnostic/logs/micro_abl_nogmsifc" "corsupcon_nogmsifc"
  run_abl "supcon_vib_dglsspp_corsupcon_nolscc" "robust_diagnostic/logs/micro_abl_nolscc" "corsupcon_nolscc"
  run_abl "supcon_vib_dglsspp_corsupcon_nocons" "robust_diagnostic/logs/micro_abl_nocons" "corsupcon_nocons"
fi

# --- Anchoring-direction micro tests (each at two settings so the direction is robust
#     to tuning): lower weight / soft alpha-blend / per-point conditioned / channel-split ---
if [ "$MODE" = "all" ] || [ "$MODE" = "abl" ] || [ "$MODE" = "anchor" ]; then
  run_abl "supcon_vib_dglsspp_corsupcon_w03" "robust_diagnostic/logs/micro_anchor_w03" "corsupcon_w03"
  run_abl "supcon_vib_dglsspp_corsupcon_w05" "robust_diagnostic/logs/micro_anchor_w05" "corsupcon_w05"
  run_abl "supcon_vib_dglsspp_corsupcon_blend03" "robust_diagnostic/logs/micro_anchor_blend03" "corsupcon_blend03"
  run_abl "supcon_vib_dglsspp_corsupcon_blend05" "robust_diagnostic/logs/micro_anchor_blend05" "corsupcon_blend05"
  run_abl "supcon_vib_dglsspp_corsupcon_cond" "robust_diagnostic/logs/micro_anchor_cond" "corsupcon_cond"
  run_abl "supcon_vib_dglsspp_corsupcon_ch64" "robust_diagnostic/logs/micro_anchor_ch64" "corsupcon_ch64"
  run_abl "supcon_vib_dglsspp_corsupcon_ch96" "robust_diagnostic/logs/micro_anchor_ch96" "corsupcon_ch96"
fi

echo "=== [dglss] plain-DGLSS base per-class check (eval-only) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
  --method supcon_vib_dglss --path robust_diagnostic/logs/supcon_vib_dglss \
  --label dglss_micro \
  2>&1 | tee "logs/dglss_base_micro_diag.log" || fail "dglss base check"

echo "All done."
