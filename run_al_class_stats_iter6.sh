#!/usr/bin/env bash
# run_al_class_stats_iter6.sh: Iteration 6 -- COVARIANCE-SPACE LOCALIZATION.
# Resolves H1 (whitening destroys each class direction) vs H2 (per-class
# directions useful, global residual alignment misleading) vs H3 (whitening
# amplifies irrelevant eigendirections).
#
# A per-class decoder alignment (the decisive test)
# B pairwise decoder alignment (w_a - w_b)
# C covariance eigen-spectrum of v_c vs Delta_mu_c
# D fractional whitening Sigma^-beta
# E rank sweep alignment + decoder gc(r)
# F per-class scalar correction
# G per-class residual decomposition (mean-shift vs total)
# H corruption control in MEAN space then decode
# I alternative pseudo-means (soft/tta/highconf/core) decoder alignment
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_class_stats_iter6.sh 3
#   SMOKE=1   bash run_al_class_stats_iter6.sh 3
#   bash run_al_class_stats_iter6.sh 3
#   CONDS="fog,crosstalk" bash run_al_class_stats_iter6.sh 3
#
# Output:
#   robust_diagnostic/logs/al_class_stats_iter6_{dglsspp,covshift_ep10}.json
#   logs/al_class_stats_iter6_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Class-stats Iteration 6 (covariance-space localization) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --k_classes 3 --k_eig 256 --rank_sweep 8,32,128 --beta_sweep 0,0.5,1.0 --alpha_sweep 0,1.0,2.0 --rho_sweep 0,0.5 --tta_augs 3"
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
  logf="logs/al_class_stats_iter6_${label}.log"
  outjson="robust_diagnostic/logs/al_class_stats_iter6_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_class_stats_iter6_diag.py \
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
  echo "=== CLASS-STATS ITERATION 6 OK ==="
  echo "  A per-class align >> 0 -> direction survives decoder geometry (H2)"
  echo "  D/E fractional/rank good, full bad -> over-whitening (H3)"
  echo "  A ~ 0 even per class -> close the line (H1)"
else
  echo "=== CLASS-STATS ITERATION 6 FAILED ==="
  exit 1
fi
