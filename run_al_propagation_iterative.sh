#!/usr/bin/env bash
# run_al_propagation_iterative.sh: make MORE use of the labeled points on the
# propagated-mean decoder (DGLSS++ fog/crosstalk only). The last design lever
# after targeted acquisition and composition both closed: single-pass -> loop.
#
# Arms (same labeled set = random anchors):
#   gcP                 baseline (current method)
#   gcA1                true means x prop counts (the mean bottleneck ceiling)
#   gc_anc_only         anchors-only means at C_prop (assignment-noise-free)
#   iter_hard           self-training loop, conf-gated replace, tau in {0.8,0.95}
#   iter_soft           soft pseudo-label weighted means at tau 0.9
#   shrink              toward-CLEAN means (1-a) M_prop + a M0, a in {0.25,0.5}
#
# Decisive:
#   iter gc grows across rounds   -> the loop is the method (grow toward
#                                    W_mean_oracle +0.72/+0.99)
#   iter gc plateaus at gcP       -> single-pass is not the bottleneck (closed)
#   gc_anc_only > gcP             -> propagation assignment noise hurts; use the
#                                    labeled points directly
#   shrink a>0 > gcP              -> the clean-means prior helps
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_iterative.sh 3
#   SMOKE=1   bash run_al_propagation_iterative.sh 3
#   bash run_al_propagation_iterative.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_iterative.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_iterative_dglsspp.json
#   logs/al_propagation_iterative_dglsspp.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Iterative self-training (DGLSS++ only) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
EXTRACTORS="$DGLSSPP"

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 2 --k_rounds 2 --tau_sweep 0.8"
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
  logf="logs/al_propagation_iterative_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_iterative_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_iterative_diag.py \
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
  echo "=== ITERATIVE SELF-TRAINING OK ==="
  echo "  iter gc grows across rounds      -> the loop is the method"
  echo "  iter gc plateaus at gcP          -> the single-pass design is closed"
  echo "  gc_anc_only > gcP                -> use the labeled points directly"
  echo "  shrink a>0 > gcP                 -> the clean-means prior helps"
else
  echo "=== ITERATIVE SELF-TRAINING FAILED ==="
  exit 1
fi
