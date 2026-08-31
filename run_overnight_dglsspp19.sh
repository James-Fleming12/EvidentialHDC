#!/usr/bin/env bash
# run_overnight_dglsspp19.sh: ONE overnight run = full DGLSS++ training on
# GeoID's 19-class map + FULL evaluation (zero-shot AND labeled ceiling) over
# every condition (docs/lin_probe_training/validation.md).
#
# Step 1 (train):  supcon_vib_dglsspp on config/labels/semantic-kitti-19.yaml
#                  (default 24 epochs at 100% data, ~10h; seq 08 never trained).
# Step 2 (eval):   map19 full-harness eval over all 8 conditions (heavy) with
#                  zero-shot (clean-fit) AND ceiling (400k corrupted-pool fit)
#                  decoders: linear / proto / raw-128-d-linear at both levels.
#
# Timing (one GPU): train ~10h (24ep) + eval ~2.5h (per cond: pool reservoir +
# full decode for zs+ceiling) = ~12.5h, under 16h.
#
# Usage:
#   DRY_RUN=1 bash run_overnight_dglsspp19.sh 3
#   SMOKE=1   bash run_overnight_dglsspp19.sh 3       # 1ep/10% train + tiny eval
#   bash run_overnight_dglsspp19.sh 3                 # the full overnight run
#   EPOCHS=18 bash run_overnight_dglsspp19.sh 3       # shorter train
#   SEVS="light,moderate,heavy" bash run_overnight_dglsspp19.sh 3   # 3-sev eval
#
# Output:
#   robust_diagnostic/logs/supcon_vib_dglsspp_19cls/SENet_valid_best  (trained)
#   robust_diagnostic/logs/lp_three_decoder_dglsspp_19cls_ceiling.json (eval)

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-24}"
SEVS="${SEVS:-heavy}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/supcon_vib_dglsspp_19cls}"
OUTJSON="${OUTJSON:-robust_diagnostic/logs/lp_three_decoder_dglsspp_19cls_ceiling.json}"
SM_FRAMES="${SM_FRAMES:-30}"

echo "OVERNIGHT DGLSS++-19: train + full zero-shot/ceiling eval"
echo "  GPU $GPU | epochs=$EPOCHS | sevs=$SEVS | log=$LOG_DIR | out=$OUTJSON"

TRAIN_ARGS="--epochs $EPOCHS"
if [ "$SMOKE" = "1" ]; then
  TRAIN_ARGS="--epochs 1 --cutoff 0.1"
fi
EVAL_ARGS="--map19 --ceiling"
if [ "$SMOKE" = "1" ]; then
  EVAL_ARGS="$EVAL_ARGS --max_frames $SM_FRAMES --clean_fit_n 5000 --pool_cap 3000"
fi

FAIL=false

echo ""
echo "======================================================"
echo "=== STEP 1: train DGLSS++ on the 19-class map ==="
echo "======================================================"
TCMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/retrain_dglsspp_19cls.py \
  --method supcon_vib_dglsspp --log_dir \"$LOG_DIR\" $TRAIN_ARGS"
echo "  CMD: $TCMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
else
  if ! eval "$TCMD" 2>&1 | tee "logs/overnight_dglsspp19_train.log"; then
    echo "ERROR: training failed" >&2
    FAIL=true
  fi
fi

echo ""
echo "======================================================"
echo "=== STEP 2: full zero-shot + ceiling eval (map19, all conds) ==="
echo "======================================================"
ECMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/lp_three_decoder_diag.py \
  --path_b \"$LOG_DIR\" --method_b supcon_vib_dglsspp --label dglsspp_19cls \
  --conds \"$CONDS\" --sevs \"$SEVS\" $EVAL_ARGS --out \"$OUTJSON\""
echo "  CMD: $ECMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
else
  if ! eval "$ECMD" 2>&1 | tee "logs/overnight_dglsspp19_eval.log"; then
    echo "ERROR: eval failed" >&2
    FAIL=true
  fi
fi

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$FAIL" = true ]; then
  echo "=== OVERNIGHT DGLSSPP-19 FAILED ==="
  exit 1
fi
echo "=== OVERNIGHT DGLSSPP-19 DONE ==="
echo "  checkpoint -> $LOG_DIR"
echo "  results    -> $OUTJSON (zero-shot + labeled ceiling, all conditions)"
