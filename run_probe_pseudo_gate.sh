#!/usr/bin/env bash
# Pseudo-label gating for the Nystrom+CG probe update: re-test the standard gates
# (conf/margin/norm/uncertainty/prior) under the NEW learned-probe decoder, plus
# diagnostics for where methods go wrong (gate AUROC for correct-vs-wrong pseudo-
# labels, per-class pseudo accuracy, retain-vs-precision, wrong-label profile).
#
# Usage:
#   bash run_probe_pseudo_gate.sh 3                # ep10+ep21, wet_ground,fog
#   bash run_probe_pseudo_gate.sh 3 "fog" ep10

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

run_gate() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [pseudo_gate] $label [$CONDS]: gates on the Nystrom+CG probe update ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_pseudo_gate_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/probe_pseudo_gate_${label}.json" \
    2>&1 | tee "logs/probe_pseudo_gate_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_gate "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_gate "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== PSEUDO-GATE OK ==="
  echo "Check logs/probe_pseudo_gate_{covshift_ep10,covshift_ep21}.log:"
  echo "  Pipelines: does any gate (conf/margin/norm/uncer) climb from no_gate toward"
  echo "  the oracle ceiling under the Nystrom+CG probe update?"
  echo "  DIAG: auroc (can the signal separate correct from wrong pseudo-labels?),"
  echo "  per_class_pseudo_acc, retain_vs_precision, wrong_profile."
else
  echo "=== PSEUDO-GATE FAILED ==="
  exit 1
fi
