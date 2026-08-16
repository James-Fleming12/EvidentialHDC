#!/usr/bin/env bash
# Quick efficiency benchmark: R1 (distance-to-prototype) vs R4 (linear probe on the
# HDC code) on the cov-shift ep10/ep21 weights. Measures fit / decode / pool-refit
# wall-clock + peak RSS. Eval-only, ~10 min total.
#
# Usage:
#   bash run_hdc_rule_bench.sh 3

set -u
set -o pipefail
GPU="${1:-3}"
echo "Using GPU $GPU"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_bench() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [bench] $label: R1 vs R4 efficiency ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/hdc_rule_bench.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" \
    --cond wet_ground \
    --out "robust_diagnostic/logs/hdc_rule_bench_$label.json" \
    > "logs/hdc_rule_bench_$label.log" 2>&1 || fail "$label"
}

run_bench "$EP10_CKPT" "covshift_ep10"
run_bench "$EP21_CKPT" "covshift_ep21"

echo ""
if [ "$FAIL" = false ]; then
  echo "=== BENCH OK ==="
  echo "Compare logs/hdc_rule_bench_{covshift_ep10,covshift_ep21}.log:"
  echo "  clean_fit_s:   R4 (LogisticRegression 10k-d) vs R1 (prototype mean)"
  echo "  decode_s:      R4 predict vs R1 cosine argmax (per val frame)"
  echo "  pool_refit_s:  the per-condition adaptation cost of each rule"
  echo "  mIoU check confirms the comparison is on the same quality result."
else
  echo "=== BENCH FAILED ==="
  exit 1
fi
