#!/usr/bin/env bash
# run_al_residual.sh: C20 -- the residual-compressibility diagnostic.
# Does the oracle residual R = W* - W0 (oracle probe - frozen probe) live in a
# low-rank subspace? If yes, AL should estimate a small residual correction
# (W = W0 + U_r C) instead of a full probe -- the C21 direction.
# Eval-only, ~1 min per condition. Runs on cov-shift ep10/ep21 (the extractors
# we want to keep) plus the ball/spec medium checkpoints (the AL-friendly ones)
# for the cross-extractor comparison.
#
# Usage:
#   bash run_al_residual.sh 3                       # cov-shift ep10+ep21 + ball/spec
#   bash run_al_residual.sh 3 covshift              # cov-shift only
#   bash run_al_residual.sh 3 ballspec              # ball/spec only

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
  echo "=== [residual] $tag [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_residual_diag.py \
    --path_b "$ckpt" --method_b "$method" --label "$tag" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_residual_$tag.json" \
    2>&1 | tee "logs/al_residual_$tag.log" || fail "$tag"
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
  echo "=== AL-RESIDUAL OK ==="
  echo "Compare logs/al_residual_{covshift_ep10,covshift_ep21,ball,spec}.log:"
  echo "  - is the ORACLE RESIDUAL CURVE mIoU(W0+R_r) steep (r=4-8 -> near oracle)?"
  echo "  - is cum_energy(r=8) ~ 0.9+ (residual compressible)?"
  echo "  - does the cov-shift residual look DIFFERENT from ball/spec's (the C19"
  echo "    explanation: cov-shift consumed the residual, ball/spec kept it) ?"
  echo "  - feat-shift effective rank: does the corruption live in a small"
  echo "    subspace of the 128-d features?"
  echo "If yes -> C21: W = W0 + U_r C with r << d, AL estimates only C."
else
  echo "=== AL-RESIDUAL FAILED ==="
  exit 1
fi
