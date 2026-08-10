#!/usr/bin/env bash
# Run the three DGLSS++ robustness-variant micro diagnostics (SupCon / class-balance /
# VIB) plus the plain-DGLSS base per-class check on a single GPU, tee-ing each step
# to its own log.
#
# Usage:
#   bash run_variant_micro.sh            # GPU 3
#   bash run_variant_micro.sh 0          # GPU 0
#
# Each training also runs the full isotropy eval (prints + saves isotropy_results.json
# into the variant's log_dir). After training, the scale_gap per-class autopsy runs for
# the same checkpoint. Logs go to logs/dglsspp_<variant>_micro.log (train + isotropy)
# and logs/dglsspp_<variant>_micro_diag.log (per-class autopsy).

set -u

GPU="${1:-3}"
echo "Using GPU $GPU"

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

for name in supcon bal vib; do
  run_train "$name"
  run_eval "$name"
done

echo "=== [dglss] plain-DGLSS base per-class check (eval-only) ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
  --method supcon_vib_dglss --path robust_diagnostic/logs/supcon_vib_dglss \
  --label dglss_micro \
  2>&1 | tee "logs/dglss_base_micro_diag.log" || fail "dglss base check"

echo "All done."
