#!/usr/bin/env bash
# run_train_geoid.sh: train + evaluate the GeoID-loss port (`supcon_vib_geoid`).
#
# Ports GeoID's feature-extractor training loss (exp_geoid.py) onto our default
# SENet backbone (no MinkowskiNet):
#   L = L_seg (semantic CE) + geo_w * L_geoid (BCE inlier-discrimination)
# The GeoID head is a 1x1 conv on the bottleneck (geoid_head=True in the
# twobranch config); synthetic displaced points are injected in the augmented
# view (geoid_displace, range-image version). Trained via isotropy_diag with
# the standard supcon_vib + VIB recipe + the GeoID BCE.
#
# Then evaluate in our R4 setup (frozen + ceiling on KITTI-C fog/crosstalk) with
# probe_linear_prop_diag, so we can compare against hyper/dgl/cov directly.
#
# Usage:
#   DRY_RUN=1 bash run_train_geoid.sh 2
#   SMOKE=1   bash run_train_geoid.sh 2
#   bash run_train_geoid.sh 2
#   EPOCHS=21 CUTOFF=1.0 bash run_train_geoid.sh 2  # full (default)
#   EPOCHS=8 CUTOFF=0.1 bash run_train_geoid.sh 2    # micro
#   GEO_W=1.0 bash run_train_geoid.sh 2             # GeoID loss weight
#
# Output:
#   robust_diagnostic/logs/geoid_full/supcon_vib_geoid/  (checkpoint)
#   robust_diagnostic/logs/probe_geoid.json               (R4 eval)

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-21}"
CUTOFF="${CUTOFF:-1.0}"
GEO_W="${GEO_W:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
SM_FRAMES="${SM_FRAMES:-10}"
METHOD="supcon_vib_geoid"
LOG_DIR="robust_diagnostic/logs/geoid_full"
CKPT_DIR="$LOG_DIR/$METHOD"
echo "GeoID-loss training | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (${EPOCHS}ep/${CUTOFF}, geo_w=$GEO_W)"

SM_FRAMES="${SM_FRAMES:-10}"

rc=0
run_phase() {
  local name="$1"; shift
  local env_pre="$1"; shift
  echo ""
  echo "==================== $name ===================="
  echo "  CMD: $env_pre $*"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] not executed"
    return 0
  fi
  local logf="logs/geoid_${name}.log"
  if [ "$SMOKE" = "1" ]; then
    echo "  [SMOKE] executing..."
    if eval "$env_pre $*" > "$logf" 2>&1; then
      echo "  [SMOKE] $name OK"
    else
      echo "  [SMOKE] $name FAILED -- tail of $logf:"
      tail -25 "$logf"
      return 1
    fi
  else
    echo "  [RUN] full phase, logging to $logf"
    eval "unset MAX_FRAMES CONDS; $env_pre $*" > "$logf" 2>&1
    local c=$?
    echo "  [$name] exit=$c"
    if [ $c -ne 0 ]; then
      echo "  WARNING: $name failed, tail of $logf:"
      tail -25 "$logf"
    fi
    return $c
  fi
}

# --- P1: train supcon_vib_geoid (full overnight by default, micro on SMOKE) ---
_ep="$EPOCHS"; _cut="$CUTOFF"; _log="$LOG_DIR"
if [ "$SMOKE" = "1" ]; then _ep="1"; _cut="0.01"; _log="${LOG_DIR}_smoke"; fi
_env="CUDA_VISIBLE_DEVICES=$GPU GEO_W=$GEO_W"
_cmd="uv run python robust_diagnostic/isotropy_diag.py --methods $METHOD --epochs $_ep --cutoff $_cut --log_dir $_log"
run_phase "p1_train_geoid" "$_env" "$_cmd" || rc=1

# --- P2: evaluate in our R4 setup (frozen + ceiling on fog/crosstalk) ---
_ckpt="$CKPT_DIR"
if [ "$SMOKE" = "1" ]; then _ckpt="${LOG_DIR}_smoke/$METHOD"; fi
_env="CUDA_VISIBLE_DEVICES=$GPU"
_cmd="uv run python robust_diagnostic/probe_linear_prop_diag.py --max_frames $MAX_FRAMES --conds fog,crosstalk --extractors \"geoid:$METHOD:$_ckpt\" --out robust_diagnostic/logs/probe_geoid.json"
run_phase "p2_eval_geoid" "$_env" "$_cmd" || rc=1

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full run with: bash run_train_geoid.sh $GPU"
  exit 0
fi
echo "=== GEOID-LOSS TRAIN + EVAL DONE ==="
echo "Checkpoint: $CKPT_DIR"
echo "R4 eval:    robust_diagnostic/logs/probe_geoid.json"
echo "Compare class_shift / frozen / ceiling vs hyper/dgl/cov (probe_linear_prop.json):"
echo "  does the GeoID inlier loss beat them in our R4 setup?"
exit 0
