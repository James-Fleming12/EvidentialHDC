#!/usr/bin/env bash
# run_train_geoid_cenet38.sh: train + evaluate the GeoID-LOSS feature extractor
# on the range-view CENET/SENet backbone SCALED to GeoID's parameter count, on
# GeoID's 19-class map (docs/lin_probe_training/validation.md).
#
# Question: does GeoID's inlier-discrimination signal work on a range-view
# network at the SAME capacity as GeoID's MinkowskiNet (MinkUNet34 ~37.85M), or
# is it specific to MinkowskiNets? Our SENet at width_mult=2.4 is 38.0M.
#
# Step 1 (train):  supcon_vib_geoid on config/arch/senet-2048p-w38.yml (38.0M)
#                  + config/labels/semantic-kitti-19.yaml. L = L_seg + GEO_W *
#                  L_geoid (BCE inlier discrimination). Default 24 epochs at
#                  100% (~10h at w2.4; seq 08 never trained).
# Step 2 (eval):   map19 full-harness eval over all 8 conditions (heavy) with
#                  zero-shot (clean-fit) AND labeled ceiling decoders.
#
# Usage:
#   DRY_RUN=1 bash run_train_geoid_cenet38.sh 3
#   SMOKE=1   bash run_train_geoid_cenet38.sh 3       # 1ep/10% + tiny eval
#   bash run_train_geoid_cenet38.sh 3                 # the full run
#   EPOCHS=18 bash run_train_geoid_cenet38.sh 3       # shorter
#   GEO_W=1.0 bash run_train_geoid_cenet38.sh 3       # GeoID loss weight
#
# Output:
#   robust_diagnostic/logs/supcon_vib_geoid_cenet38_19cls/SENet_valid_best
#   robust_diagnostic/logs/lp_three_decoder_geoid_cenet38_19cls_ceiling.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-24}"
CUTOFF="${CUTOFF:-1.0}"
SEVS="${SEVS:-heavy}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
GEO_W="${GEO_W:-1.0}"
METHOD="${METHOD:-geoid}"     # 'geoid' = pure seg + GeoID BCE (no SupCon/VIB); 'supcon_vib_geoid' = bundled
ARCH="config/arch/senet-2048p-w38.yml"
LOG_DIR="${LOG_DIR:-robust_diagnostic/logs/${METHOD}_cenet38_19cls}"
OUTJSON="${OUTJSON:-robust_diagnostic/logs/lp_three_decoder_${METHOD}_cenet38_19cls_ceiling.json}"
SM_FRAMES="${SM_FRAMES:-30}"
export GEO_W

echo "GeoID-loss on CENET at GeoID param count (38M), 19-class map"
echo "  GPU $GPU | method=$METHOD | epochs=$EPOCHS cutoff=$CUTOFF | geo_w=$GEO_W | arch=$ARCH | log=$LOG_DIR"
echo "  NOTE: ~1.5s/it on GPU2 -> 24ep/100% is ~31-33h (multiday). Resume-safe: re-running"
echo "  the same command after a crash continues from the saved epoch (SENet saved every epoch)."
echo "  Budget: EPOCHS=24 (~31h) | EPOCHS=16 CUTOFF=0.6 (~13h)"

TRAIN_ARGS="--arch $ARCH --epochs $EPOCHS --cutoff $CUTOFF"
SMOKE_CONDS="${SMOKE_CONDS:-fog,crosstalk}"
if [ "$SMOKE" = "1" ]; then
  # much faster smoke: ~64 steps of training + a 2-condition tiny eval
  TRAIN_ARGS="--arch $ARCH --epochs 1 --cutoff 0.02"
  CONDS="$SMOKE_CONDS"
  echo "  [SMOKE] 1 epoch @ 2% data (64 steps) + 2-condition eval"
fi
EVAL_ARGS="--arch $ARCH --map19 --ceiling"
if [ "$SMOKE" = "1" ]; then
  EVAL_ARGS="$EVAL_ARGS --max_frames $SM_FRAMES --clean_fit_n 5000 --pool_cap 3000"
fi

FAIL=false

echo ""
echo "======================================================"
echo "=== STEP 1: train $METHOD on CENET-38M (19-class) ==="
echo "======================================================"
TCMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/retrain_dglsspp_19cls.py \
  --method "$METHOD" --log_dir \"$LOG_DIR\" $TRAIN_ARGS"
echo "  CMD: $TCMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
else
  if ! eval "$TCMD" 2>&1 | tee "logs/train_geoid_cenet38.log"; then
    echo "ERROR: training failed" >&2
    FAIL=true
  fi
fi

echo ""
echo "======================================================"
echo "=== STEP 2: full zero-shot + ceiling eval (map19, all conds) ==="
echo "======================================================"
ECMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/lp_three_decoder_diag.py \
  --path_b \"$LOG_DIR\" --method_b \"$METHOD\" --label ${METHOD}_cenet38_19cls \
  --conds \"$CONDS\" --sevs \"$SEVS\" $EVAL_ARGS --out \"$OUTJSON\""
echo "  CMD: $ECMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
else
  if ! eval "$ECMD" 2>&1 | tee "logs/eval_geoid_cenet38.log"; then
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
  echo "=== GEOID-CENET38 FAILED ==="
  exit 1
fi
echo "=== GEOID-CENET38 DONE ==="
echo "  checkpoint -> $LOG_DIR"
echo "  results    -> $OUTJSON"
