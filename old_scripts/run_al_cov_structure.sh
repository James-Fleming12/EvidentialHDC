#!/usr/bin/env bash
# run_al_cov_structure.sh: the 8A gate -- is the decision-relevant covariance
# structured/predictable, plus the implementation diagnostics.
#
# A. COVARIANCE STRUCTURE: eff_rank of R_cov, rank-r oracle gc (low-rank?),
#    cross-condition basis (in the extra conditions via per-cond R_cov),
#    per-class covariance concentration.
# B. DELTA-Z* PREDICTABILITY: linear regressor from {margin, entropy, conf} to
#    the oracle logit correction delta_z*; R^2 per class + classification gain
#    of a feature-conditioned logit correction (fit pool, apply val).
# C. NULL CONTROL: real vs shuffled affine logit fit on the best pair.
# D. IMPLEMENTATION:
#    D1 per-pair oracle flips near the boundary (decision floors)
#    D2 per-class optimal shrinkage toward the pseudo-mean (consistent a_c?)
#    D3 density-core mean vs plain mean whitened error (mean the wrong summary?)
#    D4 propagated-mean error energy in high-gain whitening directions.
#
# Decisive:
#   A rank-8 ~ rank-64 gc  -> covariance is low-rank (pool basis + scalars)
#   B R^2 > 0.1 or gc > 0   -> the correction is predictable (tiny decision model)
#   C shuffled ~ real       -> the small positive is noise
#   D2 a_c consistent       -> one global shrinkage works
#   D3 core < plain         -> the mean is the wrong summary
#
# Usage:
#   DRY_RUN=1 bash run_al_cov_structure.sh 3
#   SMOKE=1   bash run_al_cov_structure.sh 3
#   bash run_al_cov_structure.sh 3
#   CONDS="fog,crosstalk" bash run_al_cov_structure.sh 3
#
# Output:
#   robust_diagnostic/logs/al_cov_structure_dglsspp.json
#   logs/al_cov_structure_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Covariance structure + predictability gate (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --rank_sweep 1,2,4,8 --k_eig 128 --n_top_pairs 3 --n_null 3 --b_norm 4"
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
  logf="logs/al_cov_structure_${label}.log"
  outjson="robust_diagnostic/logs/al_cov_structure_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_cov_structure_diag.py \
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
  echo "=== COVARIANCE STRUCTURE GATE OK ==="
  echo "  A rank-8 ~ rank-64 gc  -> covariance is low-rank (pool basis + scalars)"
  echo "  B R^2 > 0.1 or gc > 0   -> the correction is predictable (tiny model)"
  echo "  C shuffled ~ real       -> the small positive is noise"
  echo "  D2 a_c consistent       -> one global shrinkage works"
  echo "  D3 core < plain         -> the mean is the wrong summary"
else
  echo "=== COVARIANCE STRUCTURE GATE FAILED ==="
  exit 1
fi
