#!/usr/bin/env bash
# run_overnight_noiseinv.sh: diagnose WHY the Phase-24 noise-invariance attempt
# (`supcon_vib_additive`, volumetric noise injection) failed, using only BASIC
# augmentations (no training on -C datasets).
#
# Diagnostic-11 found class_shift (clean->corrupted per-class feature cosine) is
# the dominant zero-shot predictor (r=+0.776). The Phase-24 additive attempt
# targeted exactly this (reduce sensor-noise invariance) but was thrown out
# because at medium scale it lost on every condition. This run isolates WHY:
#   A) train supcon_vib (plain) and supcon_vib_additive (noise-invariance) at the
#      SAME micro scale (8 ep / 10%), so the comparison is capacity-matched.
#   B) evaluate BOTH with the linear-property probe (class_shift, fisher,
#      pre-sign margin, frozen/ceiling) on fog+crosstalk.
#
# Decisive questions:
#   1. Did additive REDUCE class_shift? (if yes, the mechanism worked and the
#      failure was elsewhere -- healthy trade / binarization / convergence)
#   2. Did the reduced shift translate to higher ZERO-SHOT in the code space?
#      (Diagnostic-11 says it should)
#   3. Did additive hurt the healthy conditions / clean baseline?
#      (the Phase-24 trade -- must be checked at the same capacity)
#
# This is a MICRO diagnostic (8 ep / 10%) -- fast, isolates the mechanism.
# A full-scale conditional/robust version would be the follow-up.
#
# Usage:
#   DRY_RUN=1 bash run_overnight_noiseinv.sh 2
#   SMOKE=1   bash run_overnight_noiseinv.sh 2
#   bash run_overnight_noiseinv.sh 2
#   EPOCHS=8 CUTOFF=0.1 bash run_overnight_noiseinv.sh 2

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-8}"
CUTOFF="${CUTOFF:-0.1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
PLAIN="supcon_vib"
ADD="supcon_vib_additive"
LOG_DIR="robust_diagnostic/logs/noiseinv_micro"
PLAIN_DIR="$LOG_DIR/$PLAIN"
ADD_DIR="$LOG_DIR/$ADD"
echo "Noise-invariance diagnostic | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (${EPOCHS}ep/${CUTOFF} cutoff)"

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
  local logf="logs/noiseinv_${name}.log"
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

# --- P1: train plain supcon_vib at micro scale ---
if [ "$RUN_P1" = "1" ]; then
  _ep="$EPOCHS"; _cut="$CUTOFF"
  if [ "$SMOKE" = "1" ]; then _ep="1"; _cut="0.01"; fi
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd=".venv/bin/python robust_diagnostic/isotropy_diag.py --methods $PLAIN --epochs $_ep --cutoff $_cut --log_dir $LOG_DIR"
  run_phase "p1_train_plain" "$_env" "$_cmd" || rc=1

  # --- P2: train additive (noise-invariance) at the SAME micro scale ---
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd=".venv/bin/python robust_diagnostic/isotropy_diag.py --methods $ADD --epochs $_ep --cutoff $_cut --log_dir $LOG_DIR"
  run_phase "p2_train_additive" "$_env" "$_cmd" || rc=1
fi

# --- P3: evaluate both with the linear-property probe (fog+crosstalk) ---
if [ "$RUN_P3" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd=".venv/bin/python robust_diagnostic/probe_linear_prop_diag.py --max_frames 200 --conds fog,crosstalk --extractors \"plain:$PLAIN:$PLAIN_DIR,additive:$ADD:$ADD_DIR\" --out robust_diagnostic/logs/probe_noiseinv.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p3_eval_noiseinv" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full run with: bash run_overnight_noiseinv.sh $GPU"
  exit 0
fi
echo "=== NOISE-INVARIANCE DIAGNOSTIC DONE ==="
echo "Checkpoints: $LOG_DIR/{$PLAIN,$ADD}"
echo "Eval:        robust_diagnostic/logs/probe_noiseinv.json"
echo "Compare class_shift / fisher / frozen of plain vs additive on fog+crosstalk:"
echo "  - did additive reduce class_shift? (mechanism worked?)"
echo "  - did the shift reduction raise code-space zero-shot? (Diagnostic-11 target?)"
echo "  - did additive hurt the healthy/clean conditions? (the Phase-24 trade)"
exit 0
