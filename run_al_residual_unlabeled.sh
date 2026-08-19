#!/usr/bin/env bash
# run_al_residual_unlabeled.sh: does an UNLABELED basis make the low-rank
# residual estimable?  C21 showed est-basis (U from W_sub) collapses; this
# tests pool-covariance and code-shift bases that need NO labels for U_r.
# Eval-only, ~1 min per condition.
#
# Usage:
#   bash run_al_residual_unlabeled.sh 3                       # all extractors
#   bash run_al_residual_unlabeled.sh 3 covshift              # cov-shift only

set -u
set -o pipefail
GPU="${1:-3}"
ONLY="${2:-all}"
CONDS="${3:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, only=$ONLY, conds=$CONDS"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }
COV_METHOD="supcon_vib_dglsspp_inputin_in_chan"
COV_EP10="robust_diagnostic/logs/ep10_$COV_METHOD/$COV_METHOD"
COV_EP21="robust_diagnostic/logs/med_$COV_METHOD/$COV_METHOD"
BALL_METHOD="supcon_vib_dglsspp_corsupcon_ball"
SPEC_METHOD="supcon_vib_dglsspp_corsupcon_spec"
BALL_CKPT="robust_diagnostic/logs/med_algeom_ball/$BALL_METHOD"
SPEC_CKPT="robust_diagnostic/logs/med_algeom_spec/$SPEC_METHOD"
run_one() {
  local ckpt="$1"; local method="$2"; local tag="$3"
  echo ""
  echo "=== [residual-unlabeled] $tag [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_residual_unlabeled_diag.py \
    --path_b "$ckpt" --method_b "$method" --label "$tag" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_residual_unlabeled_$tag.json" \
    2>&1 | tee "logs/al_residual_unlabeled_$tag.log" || fail "$tag"
}
if [ "$ONLY" = "all" ] || [ "$ONLY" = "covshift" ]; then
  run_one "$COV_EP10" "$COV_METHOD" "covshift_ep10"
  run_one "$COV_EP21" "$COV_METHOD" "covshift_ep21"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ballspec" ]; then
  run_one "$BALL_CKPT" "$BALL_METHOD" "ball"
  run_one "$SPEC_CKPT" "$SPEC_METHOD" "spec"
fi
echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-RESIDUAL-UNLABELED OK ==="
  echo "Check logs/al_residual_unlabeled_{covshift_ep10,covshift_ep21,ball,spec}.log:"
  echo "  - pool-cov r=4-8 delta > 0 on fog/wet where est-basis failed?"
  echo "  - code-shift delta vs pool-cov: which unlabeled subspace is better?"
  echo "  - k=2 vs k=8: does the label cost halve with the unlabeled basis?"
else
  echo "=== AL-RESIDUAL-UNLABELED FAILED ==="
  exit 1
fi
