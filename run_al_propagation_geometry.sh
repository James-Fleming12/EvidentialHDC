#!/usr/bin/env bash
# run_al_propagation_geometry.sh: what FEATURE-SPACE property predicts the
# per-class mean error of the propagated-mean decoder? (DGLSS++ only,
# fog/crosstalk)
#
# The decisive control per class: the CORRECT-ASSIGNMENT-ONLY mean.
#   Werr_all     whitened error of the full propagated mean (current)
#   Werr_correct whitened error of the correct-only mean
#   -> Werr_correct ~ 0, Werr_all large: CONTAMINATION (a better rule fixes it)
#   -> Werr_correct also large: INTRINSIC GEOMETRY / code-space saturation
#      (a rule cannot fix it; need a different mean estimator)
# Plus per-class geometry (intra_cos, inter_cos, mass), the contamination
# confusion source, and the contamination distance (core vs outlier).
#
# Decisive:
#   mean_Werr_correct_only ~ 0 -> the error is contamination (better rules help)
#   mean_Werr_correct_only ~ mean_Werr_all -> intrinsic geometry (rules cannot)
#   corr(assign_prec, Werr_all) negative -> contamination drives error
#   which geometry property tracks Werr_correct -> which estimator to build
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_geometry.sh 3
#   SMOKE=1   bash run_al_propagation_geometry.sh 3
#   bash run_al_propagation_geometry.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_geometry.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_geometry_dglsspp.json
#   logs/al_propagation_geometry_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Propagation geometry diagnostic (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 4 --loose_mult 2.0"
  echo "  [SMOKE] $SMOKE_ARGS"
fi

FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

IFS=',' read -ra EXS <<< "$EXTRACTORS"
for entry in "${EXS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"
  method="${rest%%|*}"; ckpt="${rest#*|}"
  echo ""
  echo "======================================================"
  echo "=== extractor: $label (method=$method, ckpt=$ckpt) ==="
  echo "======================================================"
  logf="logs/al_propagation_geometry_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_geometry_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_geometry_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    $SMOKE_ARGS --out \"$outjson\""
  echo "  CMD: $CMD"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] not executed"
    continue
  fi
  if eval "$CMD" 2>&1 | tee "$logf"; then
    echo "  [$label] OK -> $outjson"
  else
    echo "  [$label] FAILED -- tail of $logf:"
    tail -25 "$logf"
    fail "$label"
  fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$FAIL" = false ]; then
  echo "=== PROPAGATION GEOMETRY OK ==="
  echo "  mean_Werr_correct ~ 0       -> error is contamination (better rules help)"
  echo "  mean_Werr_correct ~ Werr_all -> intrinsic geometry (rules cannot fix)"
  echo "  corr(assign_prec, Werr) < 0  -> contamination drives error"
else
  echo "=== PROPAGATION GEOMETRY FAILED ==="
  exit 1
fi
