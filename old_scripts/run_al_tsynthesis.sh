#!/usr/bin/env bash
# al_tsynthesis: Iteration-7 -- estimate T, not labels. The escape from the
# coverage ceiling: the oracle needs the 17 class-wise vector sums T_c, which
# can be estimated from ALL points with soft assignments calibrated by a few
# true labels, instead of labeling the mass.
#   A. class-mean sample complexity (128-d + code space): how many random
#      points per class estimate the mean to cos ~0.9 -- decides the family.
#   B. T-synthesis ablation: 7A clean-mean / 7B shift / 7C shrink / 7B-oracle
#      ceiling / 7D soft-frozen / 7E soft+confusion-corrected / 7F shift-Q,
#      evaluated per-class cos(T_hat, T_oracle) -> cos(W, W_oracle) -> mIoU.
#
# Usage:
#   bash run_al_tsynthesis.sh 3                  # ep10+ep21, all 4 conds
#   bash run_al_tsynthesis.sh 3 "fog" ep10

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

run_ts() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_tsynthesis] $label [$CONDS]: estimate T, not labels ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_tsynthesis_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_tsynthesis_${label}.json" \
    2>&1 | tee "logs/al_tsynthesis_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_ts "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_ts "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-TSYNTHESIS OK ==="
  echo "Check logs/al_tsynthesis_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: sample complexity (128-d + code), 7A-7F t_cos/w_cos/miou,"
  echo "mass-estimation error. If 7E/7F beat 7D with 34-68 labels, the mass"
  echo "problem is solved without labeling the mass."
else
  echo "=== AL-TSYNTHESIS FAILED ==="
  exit 1
fi
