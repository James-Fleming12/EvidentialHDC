#!/usr/bin/env bash
# run_overnight_condin.sh: Diagnostic 6 -- conditional (stochastic) input-IN at
# FULL SCALE. The micro-scale result (cov_full_scale.md) showed stochastic
# input-IN training makes the gate a clean on/off (OFF recovers healthy, ON keeps
# fog/crosstalk rescue). This trains the full-scale stochastic variants and
# measures the gate at full scale on frozen AND ceiling.
#
# Modes:
#   DRY_RUN=1  print each command without running
#   SMOKE=1    actually run each phase at tiny MAX_FRAMES + fog only
#   (default)  full overnight run
#
# Phases:
#   P1 train stoch (p=0.5) full scale  : ~4h, isotropy_diag --epochs 10 --cutoff 1.0
#   P2 train stoch7 (p=0.7) full scale : ~4h (set TRAIN_STOCH7=0 to skip)
#   P3 gate-ON  full harness (stoch)   : frozen+ceiling, input-IN ON (default)
#   P4 gate-OFF full harness (stoch)   : frozen+ceiling, --gate_off 1
#
# Usage:
#   DRY_RUN=1 bash run_overnight_condin.sh 2
#   SMOKE=1   bash run_overnight_condin.sh 2
#   bash run_overnight_condin.sh 2
#   TRAIN_STOCH7=0 bash run_overnight_condin.sh 2   # p=0.5 only

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
TRAIN_STOCH7="${TRAIN_STOCH7:-1}"
RUN_GATE_ON="${RUN_GATE_ON:-1}"
RUN_GATE_OFF="${RUN_GATE_OFF:-1}"
EPOCHS="${EPOCHS:-10}"
CUTOFF="${CUTOFF:-1.0}"
BASE="supcon_vib_dglsspp_inputin_in_chan"
STOCH_DIR="robust_diagnostic/logs/full_stoch_$BASE"
STOCH7_DIR="robust_diagnostic/logs/full_stoch7_$BASE"
echo "Overnight conditional input-IN (full scale) | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  train=$RUN_TRAIN stoch7=$TRAIN_STOCH7 gate_on=$RUN_GATE_ON gate_off=$RUN_GATE_OFF ($EPOCHS ep / $CUTOFF cutoff)"

# Smoke caps
SM_FRAMES="${SM_FRAMES:-30}"
SM_CONDS="fog"

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
  local logf="logs/overnight_condin_${name}.log"
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

# --- P1: train the full-scale stochastic p=0.5 checkpoint (~4h) ---
if [ "$RUN_TRAIN" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/isotropy_diag.py --methods ${BASE}_stoch --epochs $EPOCHS --cutoff $CUTOFF --log_dir $STOCH_DIR"
  run_phase "p1_train_stoch" "$_env" "$_cmd" || rc=1
fi

# --- P2: train the full-scale stochastic p=0.7 checkpoint (~4h, optional) ---
if [ "$TRAIN_STOCH7" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/isotropy_diag.py --methods ${BASE}_stoch7 --epochs $EPOCHS --cutoff $CUTOFF --log_dir $STOCH7_DIR"
  run_phase "p2_train_stoch7" "$_env" "$_cmd" || rc=1
fi

# --- P3: full harness, input-IN ON (the conditional model's default eval mode) ---
if [ "$RUN_GATE_ON" = "1" ]; then
  _env="EXTRACTORS=\"stoch:${BASE}_stoch:$STOCH_DIR/${BASE}_stoch\" NUSC=0 BAL=1 OUT_SUFFIX=condin_on"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p3_gate_on" "$_env" "$_cmd" || rc=1
fi

# --- P4: full harness, input-IN OFF (--gate_off) ---
if [ "$RUN_GATE_OFF" = "1" ]; then
  _env="EXTRACTORS=\"stoch:${BASE}_stoch:$STOCH_DIR/${BASE}_stoch\" NUSC=0 BAL=1 OUT_SUFFIX=condin_off GATE_OFF=1"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p4_gate_off" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full overnight with: bash run_overnight_condin.sh $GPU"
  exit 0
fi
echo "=== OVERNIGHT CONDITIONAL INPUT-IN DONE ==="
echo "P1 stoch checkpoint: $STOCH_DIR/${BASE}_stoch"
echo "P2 stoch7 checkpoint: $STOCH7_DIR/${BASE}_stoch7"
echo "P3 gate-ON:  robust_diagnostic/logs/al_full_dataset_ep10_custom_condin_on.json"
echo "P4 gate-OFF: robust_diagnostic/logs/al_full_dataset_ep10_custom_condin_off.json"
echo "Compare frozen+ceiling ON vs OFF per condition (healthy should recover, fog/crosstalk rescue kept)."
exit 0
