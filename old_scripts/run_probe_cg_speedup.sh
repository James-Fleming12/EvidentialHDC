#!/usr/bin/env bash
# Speed up the matrix-free CG update: Nystrom warm-start, prototype residual +
# early-stop, Nystrom-warm-start-with-few-iters (preconditioner proxy), BF16 state,
# subsampled/minibatch CG. Iteration 7's CG-20 is the reference. Eval-only.
#
# Usage:
#   bash run_probe_cg_speedup.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_cg_speedup.sh 3 "fog" ep10

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

run_cg() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [cg_speedup] $label [$CONDS]: warm-start / residual / bf16 / subsample CG ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_cg_speedup_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_cg_speedup_${label}.json" \
    2>&1 | tee "logs/probe_cg_speedup_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_cg "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_cg "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== CG-SPEEDUP OK ==="
  echo "Check logs/probe_cg_speedup_{covshift_ep10,covshift_ep21}.log:"
  echo "  nys_warm : CG-5/10 from the Nystrom start ~= CG-20 from scratch?"
  echo "  residual : the prototype->probe correction, early-stopped (iters needed?)."
  echo "  precond  : Nystrom warm-start + CG-3/5/8 (the preconditioner proxy)."
  echo "  bf16     : BF16 state (FP32 accum) vs FP32 CG (cheaper GEMMs)."
  echo "  subsample: fresh 25k/12.5k/5k subset per iteration (stochastic CG, no coreset)."
else
  echo "=== CG-SPEEDUP FAILED ==="
  exit 1
fi
