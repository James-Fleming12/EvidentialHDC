#!/usr/bin/env bash
# run_retrain_dglsspp_19cls.sh: overnight retrain of DGLSS++ on GeoID's 19-class
# map (docs/lin_probe_training/validation.md). After training, the decoder
# harness evaluates against the new checkpoint with MAP19=1.
#
# Recipe: supcon_vib_dglsspp, 24 epochs at 100% data (the established medium
# run, robust_iterations.md). Train split from the config (0-7,9,10; valid 8).
#
# Usage:
#   DRY_RUN=1 bash run_retrain_dglsspp_19cls.sh 3
#   SMOKE=1   bash run_retrain_dglsspp_19cls.sh 3          # 1 epoch, 10% data
#   bash run_retrain_dglsspp_19cls.sh 3                    # overnight (24 ep)
#   EPOCHS=30 bash run_retrain_dglsspp_19cls.sh 3
#   LOG_DIR="robust_diagnostic/logs/dglsspp_19cls" bash run_retrain_dglsspp_19cls.sh 3
#
# Output:
#   robust_diagnostic/logs/dglsspp_19cls/SENet_valid_best  (the checkpoint)

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-24}"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/dglsspp_19cls}"
echo "DGLSS++ 19-class retrain | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  epochs=$EPOCHS log_dir=$LOG_DIR"

ARGS=""
if [ "$SMOKE" = "1" ]; then
  ARGS="--epochs 1 --cutoff 0.1"
  echo "  [SMOKE] 1 epoch, 10% data"
else
  ARGS="--epochs $EPOCHS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/retrain_dglsspp_19cls.py \
  --log_dir \"$LOG_DIR\" $ARGS"
echo "  CMD: $CMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
  exit 0
fi
if eval "$CMD" 2>&1 | tee "logs/retrain_dglsspp_19cls.log"; then
  echo "=== DGLSS++ 19-CLASS RETRAIN OK ==="
  echo "  checkpoint -> $LOG_DIR"
  echo "  then evaluate (MAP19):"
  echo "  CKPT=\"$LOG_DIR\" MAP19=1 CONDS=\"fog,crosstalk,wet_ground\" bash run_lp_three_decoder.sh 3"
else
  echo "=== DGLSS++ 19-CLASS RETRAIN FAILED ==="
  tail -25 "logs/retrain_dglsspp_19cls.log"
  exit 1
fi
