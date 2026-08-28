#!/usr/bin/env bash
# run_al_pair_damage.sh: Iteration 0c -- clean->corrupted decision-conditioned U.
#
# The gating experiment for the U-predictor head. Uses the clean/corrupted PAIRING
# (KITTI-C = per-frame corruptions of seq-08, same geometry+labels), conditioned on
# DECISION DAMAGE (clean right / corr wrong, or CE loss-gain). U constructions:
#   U_cross      left singulars of sum dx dz^T (feature displacement -> logit damage)
#   U_damage     covariance of the decision-FAILURE displacements
#   U_damage_w   loss-gain-weighted damage covariance
#   U_dx_all     all-pixel displacement covariance (weak control)
# Evaluated by align(U, U_oracle) AND the trust-region step gc-vs-rho.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_pair_damage.sh 2
#   SMOKE=1   bash run_al_pair_damage.sh 2
#   bash run_al_pair_damage.sh 2
#   CONDS="fog,crosstalk" bash run_al_pair_damage.sh 2
#
# Output:
#   robust_diagnostic/logs/al_pair_damage_{dglsspp,covshift_ep10}.json
#   logs/al_pair_damage_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk}"
R_SWEEP="${R_SWEEP:-2,4}"
RHO_SWEEP="${RHO_SWEEP:-0.05,0.1,0.2,0.4,0.8}"
B="${B:-8}"
MAX_PIXELS="${MAX_PIXELS:-40000}"
DAMAGE_SAMPLE="${DAMAGE_SAMPLE:-60000}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Pair-damage U (clean->corr decision-conditioned) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS r=$R_SWEEP b=$B"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --r_sweep 2,4 --b 8 --max_pixels 5000 --damage_sample 20000 --rho_sweep 0.1,0.4"
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
  logf="logs/al_pair_damage_${label}.log"
  outjson="robust_diagnostic/logs/al_pair_damage_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_pair_damage_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --r_sweep $R_SWEEP --rho_sweep $RHO_SWEEP --b $B --max_pixels $MAX_PIXELS \
    --damage_sample $DAMAGE_SAMPLE $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== PAIR-DAMAGE U OK ==="
  echo "  align > 0.7  -> U* recoverable from the pairing (label-free U exists)."
  echo "  align 0.3-0.7 -> learnable mapping (head / canonical adapter) is the route."
  echo "  align ~0     -> U* not in the corrupted side; only canonical-adapter training."
  echo "  best_gc: does a paired damage-conditioned U make the trust-region step work?"
else
  echo "=== PAIR-DAMAGE U FAILED ==="
  exit 1
fi
