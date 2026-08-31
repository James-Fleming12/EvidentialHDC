#!/usr/bin/env bash
# HDC-aligned update forms sweep: what each method needs to reach the ridge ceiling.
#   CG        : full dense S, CG solve (k = {5,10,30} iterations, O(d^2)/iter).
#   delta rule: no S, pure +/-1 associative addition ({alpha,epochs} convergence).
#   Nystrom   : random-sign sketch P (d x m), m = {100,500,1000,2000} holography.
# Each reported with float AND sign (quantized +-1 W) decode mIoU, update wall-clock,
# and pts/s. Eval-only, full 10000-d space, no block mask.
#
# NOTE: all timings now use torch.cuda.synchronize() so the pts/s are real GPU wall
# time (the previous run's CG/Nystrom numbers were async-understated). Delta rule
# alpha is 1/d ~ 1e-4 for +/-1 codes (the previous 5e-3 was 50x too large and
# diverged to all-class-0).
#
# Usage:
#   bash run_probe_hdc_update_forms.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_hdc_update_forms.sh 3 "fog" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-wet_ground,fog}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_forms() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [hdc_forms] $label [$CONDS]: CG / delta / Nystrom sweeps ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_hdc_update_forms_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_hdc_forms_${label}.json" \
    2>&1 | tee "logs/probe_hdc_forms_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_forms "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_forms "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== HDC-FORMS SWEEP OK ==="
  echo "Check logs/probe_hdc_forms_{covshift_ep10,covshift_ep21}.log:"
  echo "  The parameter at which each method's mIoU saturates toward the R4 ceiling"
  echo "  is its 'implementation need' (CG iters, delta alpha/epochs, Nystrom m)."
  echo "  Compare each to R1 proto mIoU and pts/s."
else
  echo "=== HDC-FORMS SWEEP FAILED ==="
  exit 1
fi
