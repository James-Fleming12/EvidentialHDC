#!/usr/bin/env bash
# run_fill_sevavg.sh: fill the MISSING severity-level runs needed to complete
# the 3-severity averages for the KITTI-C comparison tables.
#
# Currently-complete (per the sevavg + prior runs):
#   cov_ep10 KITTI-C : heavy (al_full_dataset_ep10.json) + light + moderate  -> COMPLETE
#   dglsspp KITTI-C  : heavy (al_full_dataset_ep10_custom_dglsspp.json) ONLY -> missing light/moderate
#   hyper  KITTI-C   : light + moderate (sevavg) ONLY                        -> missing heavy
#   NuScenes         : heavy + light + moderate for cov_nusc + dgl_nusc      -> COMPLETE
#
# This script fills:
#   F1 dglsspp KITTI-C light   (writes al_full_dataset_ep10_custom_light_dglsspp.json)
#   F2 dglsspp KITTI-C moderate(writes al_full_dataset_ep10_custom_moderate_dglsspp.json)
#   F3 hyper  KITTI-C heavy    (writes al_full_dataset_ep10_custom_heavy_hyper.json)
#
# Then the 3-severity average for dglsspp and hyper on KITTI-C is computable.
# (Each writes a FRESH output file; cov_ep10 dglsspp-runner naming is kept
# distinct via OUT_SUFFIX to avoid clobbering the sevavg files.)
#
# Modes:
#   DRY_RUN=1  print each command without running
#   SMOKE=1    run at tiny MAX_FRAMES + fog only, fail on error
#   (default)  full runs
#
# Usage:
#   DRY_RUN=1 bash run_fill_sevavg.sh 2
#   SMOKE=1   bash run_fill_sevavg.sh 2
#   bash run_fill_sevavg.sh 2
#   RUN_F1=0 RUN_F2=0 bash run_fill_sevavg.sh 2   # hyper heavy only

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_F1="${RUN_F1:-1}"   # dglsspp KITTI-C light
RUN_F2="${RUN_F2:-1}"   # dglsspp KITTI-C moderate
RUN_F3="${RUN_F3:-1}"   # hyper KITTI-C heavy
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
HYPER_CKPT="logs/kitti_pretrain"
echo "Fill-in severity batch | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (F1=$RUN_F1 F2=$RUN_F2 F3=$RUN_F3)"

SM_FRAMES="${SM_FRAMES:-10}"
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
  local logf="logs/overnight_fill_${name}.log"
  if [ "$SMOKE" = "1" ]; then
    echo "  [SMOKE] executing (max_frames=${SM_FRAMES}, conds=${SM_CONDS})..."
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

# --- F1: dglsspp KITTI-C light ---
if [ "$RUN_F1" = "1" ]; then
  _env="EXTRACTORS=\"dglsspp:supcon_vib_dglsspp:$DGLSSPP_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=light OUT_SUFFIX=light_dglsspp"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "f1_dglsspp_light" "$_env" "$_cmd" || rc=1
fi

# --- F2: dglsspp KITTI-C moderate ---
if [ "$RUN_F2" = "1" ]; then
  _env="EXTRACTORS=\"dglsspp:supcon_vib_dglsspp:$DGLSSPP_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=moderate OUT_SUFFIX=moderate_dglsspp"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "f2_dglsspp_moderate" "$_env" "$_cmd" || rc=1
fi

# --- F3: hyper (baseline) KITTI-C heavy ---
if [ "$RUN_F3" = "1" ]; then
  _env="EXTRACTORS=\"hyper:baseline:$HYPER_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=heavy OUT_SUFFIX=heavy_hyper"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "f3_hyper_heavy" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full fill-in with: bash run_fill_sevavg.sh $GPU"
  exit 0
fi
echo "=== FILL-IN DONE ==="
echo "F1 dglsspp light:    robust_diagnostic/logs/al_full_dataset_ep10_custom_light_dglsspp.json"
echo "F2 dglsspp moderate: robust_diagnostic/logs/al_full_dataset_ep10_custom_moderate_dglsspp.json"
echo "F3 hyper heavy:      robust_diagnostic/logs/al_full_dataset_ep10_custom_heavy_hyper.json"
echo "Then combine with existing heavy (dglsspp) / light+moderate (hyper) for the 3-severity average."
exit 0
