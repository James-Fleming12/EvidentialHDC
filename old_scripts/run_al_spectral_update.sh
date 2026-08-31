#!/usr/bin/env bash
# al_spectral_update: Iteration-9 -- sensitivity-bounded probe updates. The
# SAME imperfect T_hat (oracle-count x random-32 means) with FIVE decoder
# parameterizations, each tested under T_hat (robustness) and T_oracle
# (ceiling retention) -- the 2x2 verdict:
#   9A fractional ridge (beta sweep) | 9B clipped ridge (gamma sweep)
#   9C frozen residual (eta sweep)   | 9D normalized residual
#   9E unstable-subspace removal (drop bottom p% eigen-directions)
# Plus the spectrum diagnostic (gain quantiles, participation rank).
#
# Usage:
#   bash run_al_spectral_update.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_spectral_update.sh 3 "fog" ep10

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

run_spec() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_spectral_update] $label [$CONDS]: sensitivity-bounded updates ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_spectral_update_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_spectral_update_${label}.json" \
    2>&1 | tee "logs/al_spectral_update_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_spec "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_spec "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-SPECTRAL-UPDATE OK ==="
  echo "Check logs/al_spectral_update_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: the 2x2 (hat w_cos / oracle w_cos) per family, spectrum"
  echo "gain quantiles. Success = hat 0.7+ AND oracle 0.9+; failure = hat up"
  echo "but oracle down."
else
  echo "=== AL-SPECTRAL-UPDATE FAILED ==="
  exit 1
fi
