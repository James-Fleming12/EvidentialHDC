#!/usr/bin/env bash
# al_spectral_update2: Iteration-10 -- the normalized spectrum + the
# beta-continuum residual. Fixes the Iteration-9 normalization confound
# (S/N, T/N, l/N leaves the ridge EXACTLY unchanged but makes gains O(1) and
# clips bind), retests 9A fractional / 9B clipped / 9E unstable-removal, and
# tests the combination the Iteration-9 data implicated: the residual family
# with the FRACTIONAL direction (W_frozen + eta(W_beta - W_frozen)), including
# the label-budget sweep (k in {8,16,32} means/class -> 128-512 labels total).
#
# Usage:
#   bash run_al_spectral_update2.sh 3                 # ep10+ep21, all 4 conds
#   bash run_al_spectral_update2.sh 3 "fog" ep10

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

run_spec2() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_spectral_update2] $label [$CONDS]: normalized spectrum + combo ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_spectral_update2_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_spectral_update2_${label}.json" \
    2>&1 | tee "logs/al_spectral_update2_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_spec2 "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_spec2 "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-SPECTRAL-UPDATE2 OK ==="
  echo "Check logs/al_spectral_update2_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: ridge(norm) validation (~1.0), 9A/9B/9E retests, 10-COMB"
  echo "per budget (k in {8,16,32}), the method's real label cost."
else
  echo "=== AL-SPECTRAL-UPDATE2 FAILED ==="
  exit 1
fi
