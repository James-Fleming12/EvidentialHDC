#!/usr/bin/env bash
# al_label_information: Iteration-3 AL diagnostic. What does ONE true label tell
# us about the decision rule? The A/B/C experiment (nearest-anchor propagation
# vs class-centroid cosine vs decision correction) as T-label precision vs
# coverage curves, plus the ridge mIoU of each expansion, the direct-sparse
# baseline (K labels, no expansion), the soft confusion matrix (estimated and
# oracle-ceiling), and confusion-stability statistics. All operations are cosine
# decodes / lookup tables / the existing ridge: no graph, no clustering, no
# diffusion.
#
# Usage:
#   bash run_al_label_information.sh 3                  # ep10+ep21, all 4 conds
#   bash run_al_label_information.sh 3 "fog" ep10

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

run_li() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [al_label_information] $label [$CONDS]: one-label information content ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_label_information_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label "$label" --conds "$CONDS" \
    --out "robust_diagnostic/logs/al_label_information_${label}.json" \
    2>&1 | tee "logs/al_label_information_${label}.log" || fail "$label"
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep10" ]; then
  run_li "$EP10_CKPT" "covshift_ep10"
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ep21" ]; then
  run_li "$EP21_CKPT" "covshift_ep21"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-LABEL-INFORMATION OK ==="
  echo "Check logs/al_label_information_{covshift_ep10,covshift_ep21}.log"
  echo "and the .json: A/B/C precision-vs-coverage curves, ridge mIoU for"
  echo "direct_sparse / A_best / B_best / C_best / C_soft_est / C_soft_ORACLE,"
  echo "confusion stability (est-vs-oracle row-cos, top oracle pairs)."
else
  echo "=== AL-LABEL-INFORMATION FAILED ==="
  exit 1
fi
