#!/usr/bin/env bash
# Prototype-alignment diagnostic: does redefining "proximity to the prototype" as
# cosine to a LEARNED prototype (the ridge W_c) reproduce the linear probe's
# decisions at prototype-decode cost? Compares:
#   class_mean    : cosine to the class-mean code (current R1).
#   W_cos_float   : cosine to the learned W_c (the proposed redefinition).
#   W_cos_sign    : cosine to sign(W_c) (integer/popcount decode).
#   probe_ref     : the probe itself (W_c . h + b_c) -- the reference.
# Reports agreement-with-probe + ceiling mIoU + decode pts/s per condition.
#
# Usage:
#   bash run_probe_prototype_alignment.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_prototype_alignment.sh 3 "fog" ep10

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

run_align() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [proto_alignment] $label [$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_prototype_alignment_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_proto_alignment_${label}.json" \
    2>&1 | tee "logs/probe_proto_alignment_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_align "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_align "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== PROTO-ALIGNMENT OK ==="
  echo "Check logs/probe_proto_alignment_{covshift_ep10,covshift_ep21}.log:"
  echo "  If W_cos_float.agreement_with_probe is high (~0.9+), the 'learned prototype'"
  echo "  cosine reproduces the probe's decisions at prototype-decode cost -- the"
  echo "  redefinition that keeps full accuracy. W_cos_sign shows the quantization cost."
else
  echo "=== PROTO-ALIGNMENT FAILED ==="
  exit 1
fi
