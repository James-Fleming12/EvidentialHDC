#!/usr/bin/env bash
# run_al_propagation_potential.sh: can AL labels + feature-space structure
# approximate the oracle means M*? A POTENTIAL-vs-IMPLEMENTATION diagnostic.
#
# A1 proximity same-labelness (128-d, oracle signal)
# A2 nearest-anchor propagation accuracy/coverage (implementation)
# A3 per-class propagation precision
# B1 propagated mean quality (mean_cos, whitened err) + decoder gc with
#    propagated vs oracle counts (the Iteration-8 count-error split)
# C1/C2 boundary finding: where does the frozen margin separate correctly?
# D1 anchors alone vs propagated pool mean (does structure multiply labels?)
#
# Verdict rule: oracle-signal-high AND implementation-low => the estimator is
# the bottleneck (potential exists). Oracle-signal-low => direction closed.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_propagation_potential.sh 3
#   SMOKE=1   bash run_al_propagation_potential.sh 3
#   bash run_al_propagation_potential.sh 3
#   CONDS="fog,crosstalk" bash run_al_propagation_potential.sh 3
#
# Output:
#   robust_diagnostic/logs/al_propagation_potential_{dglsspp,covshift_ep10}.json
#   logs/al_propagation_potential_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Propagation-potential (oracle signal vs implementation) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b_anchors 2,4 --k_knn 1,5"
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
  logf="logs/al_propagation_potential_${label}.log"
  outjson="robust_diagnostic/logs/al_propagation_potential_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_propagation_potential_diag.py \
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
  echo "=== PROPAGATION-POTENTIAL OK ==="
  echo "  A1 high, A2 low -> proximity signal real; the propagation estimator is"
  echo "                     the bottleneck (potential exists)"
  echo "  B1 gc_oracle_counts ~ gc_mean_oracle -> the COUNT error is the bottleneck"
  echo "                     (Iteration-8 hidden failure), not the means"
  echo "  D1 prop > anchor -> structure multiplies the effective labeled set"
else
  echo "=== PROPAGATION-POTENTIAL FAILED ==="
  exit 1
fi
