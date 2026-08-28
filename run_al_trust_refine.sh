#!/usr/bin/env bash
# run_al_trust_refine.sh: Iteration 0b -- sweep the coarse-basis-sensitivity fixes
# (step-side) and the U-refinement variants, with explicit efficiency costs.
#
# STEP-SIDE (make the trust-region step robust to a coarse U):
#   oracle (ref) | tangent (baseline, failed in iter0) | A_grad (labels' own
#   gradient, no U) | A_fix (label gradient projected onto tangent span) |
#   A_hybrid (projected + residual)
# U-REFINEMENT (improve U itself):
#   U_avg (average M tangent draws -> big stack -> SVD) |
#   U_windows (more provisional windows) | U_sharpen (iterative leverage re-select)
#
# Efficiency units reported per method: R (provisional fit), SVD (svd rows x 10000,
# cost scales with rows), G (label gradient). DEC (val decode) dominates all.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_trust_refine.sh 2
#   SMOKE=1   bash run_al_trust_refine.sh 2
#   bash run_al_trust_refine.sh 2
#   CONDS="fog,crosstalk" bash run_al_trust_refine.sh 2
#
# Output:
#   robust_diagnostic/logs/al_trust_refine_{dglsspp,covshift_ep10}.json
#   logs/al_trust_refine_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
R_SWEEP="${R_SWEEP:-2,4}"
B="${B:-8}"
N_WINDOWS="${N_WINDOWS:-4}"
N_AVG_DRAWS="${N_AVG_DRAWS:-8}"
N_SHARPEN_ROUNDS="${N_SHARPEN_ROUNDS:-3}"
RHO_SWEEP="${RHO_SWEEP:-0.05,0.1,0.2,0.4,0.8}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Trust-region refinement sweep | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP b=$B rho=$RHO_SWEEP | windows=$N_WINDOWS avg=$N_AVG_DRAWS sharpen=$N_SHARPEN_ROUNDS"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --b 8 --n_windows 2 --n_avg_draws 2 --n_sharpen_rounds 1 --rho_sweep 0.1,0.4"
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
  logf="logs/al_trust_refine_${label}.log"
  outjson="robust_diagnostic/logs/al_trust_refine_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_trust_refine_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --b $B --n_windows $N_WINDOWS --n_avg_draws $N_AVG_DRAWS \
    --n_sharpen_rounds $N_SHARPEN_ROUNDS --rho_sweep $RHO_SWEEP \
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
  echo "=== TRUST-REGION REFINEMENT SWEEP OK ==="
  echo "  STEP-SIDE: does A_grad / A_fix / A_hybrid beat the coarse tangent?"
  echo "  U-REFINEMENT: does U_avg / U_windows / U_sharpen fix the trust-region step?"
  echo "  EFFICIENCY: R (provisional fit) / SVD (rows x 10000) / G (label grad) per method;"
  echo "  DEC (val decode) dominates all, so method overhead is a small fraction."
else
  echo "=== TRUST-REGION REFINEMENT SWEEP FAILED ==="
  exit 1
fi
