#!/usr/bin/env bash
# run_al_class_stats_iter7.sh: Iteration 7 -- the DECISION-BOUNDARY diagnostic
# matrix (revised per Iteration 6: recoverable != decision-relevant).
#
# A exact decision decomposition per pair (mean/prior/cov) + decision agreement
# B oracle pairwise scalar test on the pool pair direction
# C boundary-local pair directions
# D covariance diagnosis (direct ||R_cov||/||R|| and cos(R_cov,R))
# F TTA decision-space displacement direction
# G tiny pairwise logit correction (oracle vs few-label)
#
# Decisive reads:
#   A cos_mean/cos_cov -> which component explains the PAIRWISE decision shift
#   dec_agree mean-oracle ~1.0 on errors -> mean-only decoder already matches
#   B gamma* helps -> pairwise direction usable; estimate gamma from labels
#   D ||R_cov||/||R|| direct test of "covariance dominated"
#   F TTA displacement predicts oracle decision movement?
#   G pairwise logit (alpha,beta) oracle vs few-label
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter7.sh 3
#   SMOKE=1   bash run_al_class_stats_iter7.sh 3
#   bash run_al_class_stats_iter7.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter7.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter7_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter7_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iteration 7 (decision-boundary matrix) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --k_classes 3 --n_pairs 3 --gamma_sweep -0.5,0,1.0 --b_per_class 2 --k_eig 128 --tta_augs 4"
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
  logf="logs/al_class_stats_iter7_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter7_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter7_diag.py \
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
  echo "=== CLASS-STATS ITERATION 7 OK ==="
  echo "  A cos_mean/cos_cov -> which component explains the pairwise decision shift"
  echo "  dec_agree mean-oracle ~1.0 -> mean-only decoder already matches oracle"
  echo "  B gamma* helps -> pairwise direction usable"
  echo "  D ||R_cov||/||R|| direct covariance test"
  echo "  F TTA displacement predicts oracle decision movement?"
  echo "  G pairwise logit correction oracle vs few-label"
else
  echo "=== CLASS-STATS ITERATION 7 FAILED ==="
  exit 1
fi
