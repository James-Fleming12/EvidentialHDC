#!/usr/bin/env bash
# run_retrain_dglsspp_19cls.sh: overnight retrain of a DGLSS++-family extractor
# on GeoID's 19-class map (docs/lin_probe_training/validation.md). After
# training, the decoder harness evaluates against the new checkpoint with
# MAP19=1.
#
# Recipe: default supcon_vib_dglsspp, 24 epochs at 100% data (~25 min/epoch,
# ~10h). Set METHOD=supcon_vib_geoid to also train the GeoID-LOSS variant
# (inlier-discrimination head, ~30 min/epoch). Train split from the config
# (0-7,9,10; valid 8).
#
# Usage:
#   DRY_RUN=1 bash run_retrain_dglsspp_19cls.sh 3
#   SMOKE=1   bash run_retrain_dglsspp_19cls.sh 3          # 1 epoch, 10% data
#   bash run_retrain_dglsspp_19cls.sh 3                    # overnight dglsspp-19
#   METHOD=supcon_vib_geoid bash run_retrain_dglsspp_19cls.sh 3   # GeoID-loss variant
#   EPOCHS=18 bash run_retrain_dglsspp_19cls.sh 3          # shorter (fits two runs)
#   LOG_DIR="robust_diagnostic/logs/dglsspp_19cls" bash run_retrain_dglsspp_19cls.sh 3
#
# Two-run 16h budget (sequential): EPOCHS=16 on each -> ~6.7h + ~8h = ~14.7h.
# If a 2nd GPU is free, run both at EPOCHS=24 in parallel -> ~10h wall clock.
#
# Output:
#   $LOG_DIR/SENet_valid_best  (the checkpoint)

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-24}"
METHOD="${METHOD:-supcon_vib_dglsspp}"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/${METHOD}_19cls}"
GEO_W="${GEO_W:-1.0}"
export GEO_W
echo "DGLSS++-family 19-class retrain | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  method=$METHOD epochs=$EPOCHS geo_w=$GEO_W log_dir=$LOG_DIR"

ARGS=""
if [ "$SMOKE" = "1" ]; then
  ARGS="--epochs 1 --cutoff 0.1"
  echo "  [SMOKE] 1 epoch, 10% data"
else
  ARGS="--epochs $EPOCHS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/retrain_dglsspp_19cls.py \
  --method \"$METHOD\" --log_dir \"$LOG_DIR\" $ARGS"
echo "  CMD: $CMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
  exit 0
fi
if eval "$CMD" 2>&1 | tee "logs/retrain_${METHOD}_19cls.log"; then
  echo "=== RETRAIN OK ($METHOD) ==="
  echo "  checkpoint -> $LOG_DIR"
  echo "  then evaluate (MAP19):"
  echo "  METHOD=\"$METHOD\" CKPT=\"$LOG_DIR\" MAP19=1 CONDS=\"fog,crosstalk,wet_ground\" bash run_lp_three_decoder.sh 3"
else
  echo "=== RETRAIN FAILED ($METHOD) ==="
  tail -25 "logs/retrain_${METHOD}_19cls.log"
  exit 1
fi

