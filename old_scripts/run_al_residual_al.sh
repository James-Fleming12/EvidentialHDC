#!/usr/bin/env bash
# run_al_residual_al.sh: C21 -- sparse-label estimation of the low-rank
# residual correction W = W0 + U_r C on the cov-shift + ball/spec extractors.
# C20 showed the oracle residual is low-rank (eff-rank 4-5, r=8 == oracle).
# C21 tests: can a cheap label budget estimate the 17 x r coefficients C, and
# does the low-rank residual AL beat the full-probe estimation (Ite-7/8 fail)?
# Eval-only, ~1 min per condition.
#
# Usage:
#   bash run_al_residual_al.sh 3                       # cov-shift ep10+ep21 + ball/spec
#   bash run_al_residual_al.sh 3 covshift              # cov-shift only
#   bash run_al_residual_al.sh 3 ballspec              # ball/spec only

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
  echo "=== [residual-AL] $tag [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_residual_al_diag.py \
    --path_b "$ckpt" --method_b "$method" --label "$tag" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_residual_al_$tag.json" \
    2>&1 | tee "logs/al_residual_al_$tag.log" || fail "$tag"
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
  echo "=== AL-RESIDUAL-AL OK ==="
  echo "Compare logs/al_residual_al_{covshift_ep10,covshift_ep21,ball,spec}.log:"
  echo "  - oracle-basis r=8: does the low-rank C beat frozen on the hard"
  echo "    conditions (fog/wet) where the full-probe AL failed (C18) ?"
  echo "  - est-basis vs oracle-basis: can labels discover the directions?"
  echo "  - vs full_probe: does residual-subspace AL beat T_hat estimation?"
  echo "If oracle-basis r=8 is positive on fog/wet -> C21 works; if est-basis"
  echo "matches -> it is deployable without any oracle basis."
else
  echo "=== AL-RESIDUAL-AL FAILED ==="
  exit 1
fi
