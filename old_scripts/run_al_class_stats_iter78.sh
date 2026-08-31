#!/usr/bin/env bash
# run_al_class_stats_iter78.sh: Iterations 7.5 + 8 in ONE run.
#
# PART A (7.5) covariance-only DECODER ceiling: W0 / mean / prior / cov /
#   mean+prior / mean+cov / full oracle, each with gc + dec_agree (overall and
#   on errors). R_cov_frac = ||W* - W_mean_oracle||/||R|| (independent).
# PART B (8) pairwise LOGIT correction ceiling-first:
#   B1 per-pair oracle, B2 label-estimation (1..16 labels), B3 random pairs,
#   B4 boundary-conditioned, B5 global per-class bias, B6 shared scalar.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter78.sh 3
#   SMOKE=1   bash run_al_class_stats_iter78.sh 3
#   bash run_al_class_stats_iter78.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter78.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter78_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter78_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iterations 7.5+8 (covariance ceiling + logit correction) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --n_pairs 3 --b_labels 2,4 --tau 1.0"
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
  logf="logs/al_class_stats_iter78_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter78_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter78_diag.py \
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
  echo "=== CLASS-STATS ITERATIONS 7.5+8 OK ==="
  echo "  cov_only gc ~ oracle gc + dec_agree ~ 1 -> covariance IS the problem"
  echo "  B1 oracle per-pair gc -> which pairs benefit"
  echo "  B2 few-label tracks oracle -> the logit correction is estimable"
  echo "  B6 shared scalar -> tiny parameter count the key test"
else
  echo "=== CLASS-STATS ITERATIONS 7.5+8 FAILED ==="
  exit 1
fi
