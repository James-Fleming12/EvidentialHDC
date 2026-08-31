#!/usr/bin/env bash
# run_al_shared_basis.sh: Iteration 1 -- do the residuals across conditions share
# a usable structure?
#
# The canonical adapter and efficient bank both assume ONE low-rank structure U0
# serves ALL corruption conditions (R_c ~ U0 C_c). This measures whether the fog /
# crosstalk / snow / wet_ground residuals actually live in the SAME directions:
#   - per-condition capture of the POOLED basis (ratio vs the condition's own)
#   - pairwise subspace agreement between conditions
#
# The answer gates the whole roadmap:
#   PASS (ratio ~1 everywhere, pairwise cos high) -> one shared basis; a single
#        adapter (canonical U0 / one bank U) is well-posed.
#   FAIL (a 'left-out' condition has low pool/own ratio) -> the shared-adapter
#        assumption is violated; the canonical adapter is structurally impossible
#        and the bank must be condition-specific.
#
# Runs BOTH DGLSS++ (primary) and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_shared_basis.sh 2
#   SMOKE=1   bash run_al_shared_basis.sh 2
#   bash run_al_shared_basis.sh 2
#   CONDS="fog,crosstalk" bash run_al_shared_basis.sh 2
#
# Output:
#   robust_diagnostic/logs/al_shared_basis_{dglsspp,covshift_ep10}.json
#   logs/al_shared_basis_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4,8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Shared-basis (residual subspace) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4"
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
  logf="logs/al_shared_basis_${label}.log"
  outjson="robust_diagnostic/logs/al_shared_basis_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_shared_basis_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== SHARED-BASIS OK ==="
  echo "  ratio ~1 everywhere + high pairwise cos -> one shared basis, single adapter is well-posed."
  echo "  a 'left-out' condition (low ratio) -> shared-adapter assumption violated."
  echo "  effective_rank_pooled -> the r to use for the shared structure."
else
  echo "=== SHARED-BASIS FAILED ==="
  exit 1
fi
