#!/usr/bin/env bash
# regate_algeom_beta_eta.sh: re-sweep (beta, eta) of the Iteration-10 fractional-
# residual AL update on the NEW medium feature spaces (ball/spec), at the CHEAP
# k=8 budget (64-72 labels total). The default beta=0.75 / eta=0.1 were tuned on
# the base's STEEPER spectrum; the AL-geometry objectives flattened it (prank
# 3-5 vs 2-3), so the optimal (beta, eta) may have moved. This answers: does the
# ball/spec feature space have AL headroom (esp. snow/wet_ground) that the
# default (beta, eta) missed? Eval-only, minutes per condition.
#
# Usage:
#   bash regate_algeom_beta_eta.sh 3                       # ball+spec, final+valid_best
#   bash regate_algeom_beta_eta.sh 3 "fog,crosstalk"       # subset of conditions

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${2:-fog,crosstalk,snow,wet_ground}"
echo "Using GPU $GPU, conds=$CONDS"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

run_resweep() {
  local method="$1"; local label="$2"; local tag="$3"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  if [ "$tag" = "valid_best" ]; then
    ckpt_dir="$ckpt_dir/valid_best"
    if [ ! -f "$ckpt_dir/SENet" ]; then
      echo "=== [$label:$tag] SKIP: no SENet copy ==="
      return 0
    fi
  fi
  local out_tag="${label}_${tag}"
  echo ""
  echo "=== [beta-eta re-sweep] $out_tag [$CONDS] on $method (k=8) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_betaeta_resweep.py \
    --path_b "$ckpt_dir" --method_b "$method" --label "$out_tag" --conds "$CONDS" \
    --mean_k 8 \
    --out "robust_diagnostic/logs/al_betaeta_med_$out_tag.json" \
    2>&1 | tee "logs/al_betaeta_med_$out_tag.log" || fail "$out_tag"
}

run_resweep "$BALL" "ball" "final"
run_resweep "$SPEC" "spec" "final"
run_resweep "$BALL" "ball" "valid_best"
run_resweep "$SPEC" "spec" "valid_best"

echo ""
if [ "$FAIL" = false ]; then
  echo "=== BETA-ETA RE-SWEEP OK ==="
  echo "For each arm/checkpoint, logs/al_betaeta_med_{ball,spec}_{final,valid_best}.log"
  echo "give the 10-COMB delta (combo - frozen) across beta x eta at k=8 (64-72"
  echo "labels). Answer:"
  echo "  - is there a (beta,eta) that makes snow/wet_ground combo POSITIVE (or"
  echo "    least-negative) that the default beta=0.75/eta=0.1 missed?"
  echo "  - does the best (beta,eta) differ from 0.75/0.1 (i.e. did the spectrum"
  echo "    flattening move the AL optimum)?"
  echo "  If yes on either -> continue training; if snow/wet negative across the"
  echo "  whole grid -> the negative-AL is the T_hat/rare-class issue (Iteration"
  echo "  11), not a (beta,eta) mismatch."
else
  echo "=== BETA-ETA RE-SWEEP FAILED ==="
  exit 1
fi
