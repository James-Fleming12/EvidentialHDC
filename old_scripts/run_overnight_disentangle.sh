#!/usr/bin/env bash
# run_overnight_disentangle.sh: combined overnight --
#   1. Train the HyperLiDAR feature extractor (plain supervised `baseline`) on
#      NuScenes, for tomorrow's GeoID-style comparisons (~5h at 21ep).
#   2. Disentangle WHAT training component raises the fog/crosstalk ceiling:
#      train two cov-shift variants at 6ep/100% KITTI and measure their
#      frozen/ceiling vs the existing `_in_chan` checkpoint:
#        - supcon_vib_dglsspp_inputin    (input-IN only, BN trunk)  -> isolates internal-IN
#        - supcon_vib_dglsspp_inputin_in (input-IN + internal IN, all channels)
#                                         -> isolates the channel-restriction {0,4}
#      Reference: supcon_vib_dglsspp_inputin_in_chan (the cov-shift winner, ep10 exists).
#   3. Run the fog/crosstalk ceiling battery on the new variants (same probe as the
#      full-scale harness: frozen + ceiling), so the overnight yields numbers, not
#      just checkpoints.
#
# Modes:
#   DRY_RUN=1  print each command without running
#   SMOKE=1    actually run each phase at tiny settings, fail on error
#   (default)  full overnight run
#
# Usage:
#   DRY_RUN=1 bash run_overnight_disentangle.sh 2
#   SMOKE=1   bash run_overnight_disentangle.sh 2
#   bash run_overnight_disentangle.sh 2
#   RUN_HYPER=0 bash run_overnight_disentangle.sh 2   # skip HyperLiDAR training

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_HYPER="${RUN_HYPER:-1}"
RUN_TRAIN_DIS="${RUN_TRAIN_DIS:-1}"
RUN_EVAL_DIS="${RUN_EVAL_DIS:-1}"

# scales
NUSC_EPOCHS="${NUSC_EPOCHS:-21}"
DIS_EPOCHS="${DIS_EPOCHS:-6}"
DIS_CUTOFF="${DIS_CUTOFF:-1.0}"

# checkpoints / log dirs
HYPER_NUSC_CKPT="logs/nusc_pretrain"
DIS_A="supcon_vib_dglsspp_inputin"
DIS_B="supcon_vib_dglsspp_inputin_in"
DIS_REF="supcon_vib_dglsspp_inputin_in_chan"
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
DIS_A_DIR="robust_diagnostic/logs/dis6_$DIS_A"
DIS_B_DIR="robust_diagnostic/logs/dis6_$DIS_B"

echo "Overnight disentangle | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE"
echo "  hyper_nusc=${RUN_HYPER} (${NUSC_EPOCHS}ep) train_dis=${RUN_TRAIN_DIS} (${DIS_EPOCHS}ep/${DIS_CUTOFF}) eval_dis=${RUN_EVAL_DIS}"

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
  local logf="logs/overnight_disentangle_${name}.log"
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

# --- P1: train HyperLiDAR (plain supervised baseline) on NuScenes ---
if [ "$RUN_HYPER" = "1" ]; then
  if [ -f "$HYPER_NUSC_CKPT/SENet" ]; then
    echo "  [P1] NuScenes HyperLiDAR checkpoint exists at $HYPER_NUSC_CKPT/SENet -- skipping"
  else
    _ep="$NUSC_EPOCHS"; _cut="1.0"; _log="$HYPER_NUSC_CKPT"
    if [ "$SMOKE" = "1" ]; then _ep="1"; _cut="0.01"; _log="${HYPER_NUSC_CKPT}_smoke"; fi
    _env="CUDA_VISIBLE_DEVICES=$GPU"
    _cmd="uv run python robust_diagnostic/train_covshift_nuscenes.py --method baseline --epochs $_ep --cutoff $_cut --log_dir $_log"
    run_phase "p1_train_hyper_nusc" "$_env" "$_cmd" || rc=1
  fi
fi

# --- P2: train disentangle variant A (input-IN only, BN trunk) ---
if [ "$RUN_TRAIN_DIS" = "1" ]; then
  _ep="$DIS_EPOCHS"; _cut="$DIS_CUTOFF"; _cond=""; _log="$DIS_A_DIR"
  if [ "$SMOKE" = "1" ]; then _ep="1"; _cut="0.01"; _cond="--conditions fog --frames 5"; _log="${DIS_A_DIR}_smoke"; fi
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/isotropy_diag.py --methods $DIS_A --epochs $_ep --cutoff $_cut --log_dir $_log $_cond"
  run_phase "p2_train_dis_a" "$_env" "$_cmd" || rc=1

  # --- P3: train disentangle variant B (input-IN + internal IN, all channels) ---
  _ep="$DIS_EPOCHS"; _cut="$DIS_CUTOFF"; _cond=""; _log="$DIS_B_DIR"
  if [ "$SMOKE" = "1" ]; then _ep="1"; _cut="0.01"; _cond="--conditions fog --frames 5"; _log="${DIS_B_DIR}_smoke"; fi
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/isotropy_diag.py --methods $DIS_B --epochs $_ep --cutoff $_cut --log_dir $_log $_cond"
  run_phase "p3_train_dis_b" "$_env" "$_cmd" || rc=1
fi

# --- P4: eval the disentangle variants (fog/crosstalk frozen+ceiling) ---
if [ "$RUN_EVAL_DIS" = "1" ]; then
  _a_dir="$DIS_A_DIR"; _b_dir="$DIS_B_DIR"
  if [ "$SMOKE" = "1" ]; then _a_dir="${DIS_A_DIR}_smoke"; _b_dir="${DIS_B_DIR}_smoke"; fi
  _env="EXTRACTORS=\"dis_a:$DIS_A:$_a_dir/${DIS_A},dis_b:$DIS_B:$_b_dir/${DIS_B},cov_ref:$DIS_REF:robust_diagnostic/logs/ep10_${DIS_REF}/${DIS_REF}\" NUSC=0 BAL=0 OUT_SUFFIX=disentangle"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p4_eval_disentangle" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full overnight with: bash run_overnight_disentangle.sh $GPU"
  exit 0
fi
echo "=== OVERNIGHT DISENTANGLE DONE ==="
echo "P1 HyperLiDAR NuScenes:  $HYPER_NUSC_CKPT/SENet"
echo "P2 dis_a (inputin):      $DIS_A_DIR/${DIS_A}"
echo "P3 dis_b (inputin_in):   $DIS_B_DIR/${DIS_B}"
echo "P4 eval:                 robust_diagnostic/logs/al_full_dataset_ep10_custom_disentangle.json"
echo "Compare dis_a/dis_b/cov_ref fog+crosstalk ceiling: isolates internal-IN and"
echo "channel-restriction contributions to the recoverable ceiling."
exit 0
