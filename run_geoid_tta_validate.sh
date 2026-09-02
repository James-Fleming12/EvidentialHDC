#!/usr/bin/env bash
# run_geoid_tta_validate.sh: does GeoID's TTA mechanism work on OUR range-view
# network? (GeoID TTA = inlier-discrimination loss + BiUPF confidence gating,
# NOT semantic pseudo-labelling). Measures the seg-head mIoU frozen vs adapted.
#
# Parameterized so it runs on a 6.8M geoid model now and on the ~38M
# geoid-cenet38 later (CKPT / METHOD / ARCH) for a direct capacity comparison.
#
# Requires a TRAINED geoid model (the geoid head is used for BiUPF and adapted):
#   - the geoid-cenet38 40h run (robust_diagnostic/logs/geoid_cenet38_19cls) once done
#   - or a 6.8M pure-geoid retrain
#
# Usage:
#   DRY_RUN=1 bash run_geoid_tta_validate.sh 3
#   SMOKE=1   bash run_geoid_tta_validate.sh 3
#   bash run_geoid_tta_validate.sh 3                       # defaults: geoid-cenet38, fog/crosstalk/wet_ground
#   CKPT=... METHOD=... ARCH=... bash run_geoid_tta_validate.sh 3
#   CONDS="fog,crosstalk,wet_ground" SEVS="heavy" TTA_FRAMES=100 TTA_STEPS=3 bash run_geoid_tta_validate.sh 3
#
# Output:
#   robust_diagnostic/logs/geoid_tta_validate.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
CONDS="${CONDS:-fog,crosstalk,wet_ground}"
SEVS="${SEVS:-heavy}"
CKPT="${CKPT:-robust_diagnostic/logs/geoid_cenet38_19cls}"
METHOD="${METHOD:-geoid}"
ARCH="${ARCH:-config/arch/senet-2048p-w38.yml}"
TTA_FRAMES="${TTA_FRAMES:-100}"
TTA_STEPS="${TTA_STEPS:-3}"
TTA_LR="${TTA_LR:-0.001}"
TAU_R="${TAU_R:-0.6}"
OUTJSON="${OUTJSON:-robust_diagnostic/logs/geoid_tta_validate.json}"
echo "GeoID-TTA validation | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  ckpt=$CKPT method=$METHOD arch=$ARCH conds=$CONDS sevs=$SEVS"
echo "  tta_frames=$TTA_FRAMES tta_steps=$TTA_STEPS tta_lr=$TTA_LR tau_r=$TAU_R"

ARGS="--path_b \"$CKPT\" --method \"$METHOD\" --arch \"$ARCH\" --conds \"$CONDS\" --sevs \"$SEVS\" \
  --tta_frames $TTA_FRAMES --tta_steps $TTA_STEPS --tta_lr $TTA_LR --tau_r $TAU_R --out \"$OUTJSON\""
if [ "$SMOKE" = "1" ]; then
  ARGS="--path_b \"$CKPT\" --method \"$METHOD\" --arch \"$ARCH\" --conds fog --sevs heavy \
  --max_frames 40 --tta_frames 20 --tta_steps 1 --tta_lr $TTA_LR --tau_r $TAU_R --val_size 5000 --out \"$OUTJSON\""
  echo "  [SMOKE] 20 TTA frames, 1 step, 40 total frames, fog only"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/geoid_tta_validate_diag.py $ARGS"
echo "  CMD: $CMD"
if [ "$DRY_RUN" = "1" ]; then
  echo "  [DRY] not executed"
  exit 0
fi
if eval "$CMD" 2>&1 | tee "logs/geoid_tta_validate.log"; then
  echo "=== GEOID-TTA VALIDATION OK -> $OUTJSON ==="
  echo "  delta > 0  -> GeoID TTA works on our range-view encoder"
  echo "  delta ~ 0  -> the inlier signal / BiUPF gate is not usable here"
else
  echo "=== GEOID-TTA VALIDATION FAILED ==="
  tail -25 "logs/geoid_tta_validate.log"
  exit 1
fi
