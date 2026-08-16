#!/usr/bin/env bash
# Separator-form ablation: can a FIRST-ORDER separator express the probe's boundary
# rotation (or is it genuinely in the cross-coordinate covariance)? Tests:
#   r1_proto     : W = mu (baseline)
#   diag_lda     : W_cj = mu_cj/(1-mu_cj^2)  -- per-class coordinate weights from
#                  FIRST-ORDER class sums only (the key test A).
#   shared_diag  : W_c = q .* mu_c, pooled q -- domain-wide rotation (O(d)).
#   perceptron   : init W=mu, correct mistakes by +/-h, batched matmul (test B).
#   passive_agg  : margin-based perceptron variant.
#   nystrom      : sketch covariance (current best cheap approx).
#   full_ridge   : X^T X solve (the ceiling).
# Reports mIoU (ceiling) + update wall-clock + statistic order per condition.
#
# Usage:
#   bash run_probe_separator_ablation.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_separator_ablation.sh 3 "fog" ep10

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

run_sep() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [separator] $label [$CONDS]: first-order separator forms ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_separator_ablation_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_separator_ablation_${label}.json" \
    2>&1 | tee "logs/probe_separator_ablation_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_sep "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_sep "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== SEPARATOR-ABLATION OK ==="
  echo "Check logs/probe_separator_ablation_{covshift_ep10,covshift_ep21}.log:"
  echo "  A. diag_lda (first-order) vs full_ridge: does coordinate reweighting capture"
  echo "     most of the R4 gain? (=> first-order stats suffice)"
  echo "  B. perceptron / passive_agg (first-order mistakes) vs full_ridge: can a"
  echo "     genuine separator be learned without covariance?"
else
  echo "=== SEPARATOR-ABLATION FAILED ==="
  exit 1
fi
