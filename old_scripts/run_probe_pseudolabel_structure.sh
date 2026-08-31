#!/usr/bin/env bash
# probe_pseudolabel_structure: the comprehensive S/T decomposition diagnostic for
# the label-free probe update (overnight). Answers, per condition:
#   A. S/T decomposition: does gating T while keeping S=all work (vs Iteration 9's
#      S_and_T gated failure)?
#   B. W decomposition: are wrong pseudo-labels noise (A), rotation (B), or is
#      correct-only T coverage-limited (C)?
#   C. influence/leverage: is a leverage-aware gate different from a confidence gate?
#   D. reliability: per-class precision spread -> global vs per-class gate.
#   E. agreement/regions: prototype-vs-probe disagreement populations + gates.
#   F. coverage: which gates preserve the covariance (frob_diff_ratio).
#   G. coverage-preserving gates: per-cluster / per-class / per-region top-conf.
#   H. oracle gate decomposition: precision -> mIoU curves per removal strategy.
#
# Usage:
#   bash run_probe_pseudolabel_structure.sh 3                  # ep10+ep21, all 4 conds
#   bash run_probe_pseudolabel_structure.sh 3 "fog,crosstalk" ep10

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
ONLY="${3:-all}"
echo "Using GPU $GPU, conds=$CONDS"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_struct() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [pseudolabel_struct] $label [$CONDS]: S/T decomposition diagnostic ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_pseudolabel_structure_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_pseudolabel_struct_${label}.json" \
    2>&1 | tee "logs/probe_pseudolabel_struct_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_struct "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_struct "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== PSEUDOLABEL-STRUCTURE OK ==="
  echo "Check logs/probe_pseudolabel_struct_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json. The st section answers the method design: S_all,T_gated vs"
  echo "S_all,T_all vs S_gated,T_gated; w_decomp gives the A/B/C diagnosis; H gives"
  echo "the precision->mIoU ceiling of T-only gating."
else
  echo "=== PSEUDOLABEL-STRUCTURE FAILED ==="
  exit 1
fi
