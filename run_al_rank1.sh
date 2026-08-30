#!/usr/bin/env bash
# run_al_rank1.sh: is the rank-1-per-label decomposition of the update viable?
# A DIAGNOSTIC (not a black box): tests whether treating each label as its own
# rank-1 update direction can work, isolating each potential failure:
#   D1/D2 atomic: is an individual rank-1 update aligned with R and useful?
#   D3 cancelation: do the per-label directions reinforce or cancel?
#   D4 per-label scale (oracle): is the failure step-size (each label needs its
#      own eta)?
#   D5 rejection: can a label-free score pick the good rank-1 updates?
# The linearity fact is central: sequential-with-fixed-eta == aggregate, so the
# only levers that matter are per-label scale, rejection, and adaptive re-query.
#
# Runs BOTH DGLSS++ and cov-shift.
#
# Usage:
#   DRY_RUN=1 bash run_al_rank1.sh 2
#   SMOKE=1   bash run_al_rank1.sh 2
#   bash run_al_rank1.sh 2
#   CONDS="fog,crosstalk" bash run_al_rank1.sh 2
#
# Output:
#   robust_diagnostic/logs/al_rank1_{dglsspp,covshift_ep10}.json
#   logs/al_rank1_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
B="${B:-8}"
ETA="${ETA:-0.05}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Rank-1-per-label decomposition (diagnostic) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  conds=$CONDS b=$B eta=$ETA"

DGLSSPP="dglsspp|supcon_vib_dglsspp|robust_diagnostic/logs/supcon_vib_dglsspp"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
EXTRACTORS="$DGLSSPP,$COV"
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 3000 --val_size 6000 --max_clean 5000 --b 4 --eta 0.05 --cand_frac 0.2 --tta_augs 3"
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
  logf="logs/al_rank1_${label}.log"
  outjson="robust_diagnostic/logs/al_rank1_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_rank1_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    --b $B --eta $ETA $SMOKE_ARGS --out \"$outjson\""
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
  echo "=== RANK-1 DECOMPOSITION DIAGNOSTIC OK ==="
  echo "  D1/D2: are individual rank-1 updates aligned with R and useful?"
  echo "  D3: do the directions reinforce or cancel (aggregate vs mean individual)?"
  echo "  D4: is the failure step-size (oracle-scaled sequential vs aggregate)?"
  echo "  D5: can a label-free score pick the good rank-1 updates?"
  echo "  Idea WORKS if D2 has positives AND D4/D5 show the aggregate was the wrong"
  echo "  packaging. DEAD if D2 is all-negative (atomic failure)."
else
  echo "=== RANK-1 DECOMPOSITION FAILED ==="
  exit 1
fi
