#!/usr/bin/env bash
# run_overnight_sevavg.sh: compute the 3-severity average (light/moderate/heavy)
# mIoU per condition on KITTI-C and NuScenes-C, to match GeoID's reporting
# (their tables average all severity levels). Heavy already exists; this adds
# light + moderate.
#
# Modes:
#   DRY_RUN=1  print each command without running (syntax/path check)
#   SMOKE=1    actually run each phase at tiny MAX_FRAMES + fog only, fail on error
#   (default)  full overnight run
#
# Phases:
#   P1 KITTI-C light   : cov_ep10, all 8 conds, severity=light
#   P2 KITTI-C moderate: cov_ep10, all 8 conds, severity=moderate
#   P3 NuScenes-C light:  cov_nusc + dgl_nusc, all 8 conds, severity=light
#   P4 NuScenes-C moderate: cov_nusc + dgl_nusc, all 8 conds, severity=moderate
#
# Each phase writes a FRESH output file (no stale-info / skip collisions).
#
# Usage:
#   DRY_RUN=1 bash run_overnight_sevavg.sh 2
#   SMOKE=1   bash run_overnight_sevavg.sh 2
#   bash run_overnight_sevavg.sh 2
#   RUN_P1=0 RUN_P2=0 bash run_overnight_sevavg.sh 2   # NuScenes only

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_P4="${RUN_P4:-1}"
COV_CKPT="robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
COV_NUSC_CKPT="robust_diagnostic/logs/nusc_covshift_21ep"
DGL_NUSC_CKPT="robust_diagnostic/logs/nusc_dglsspp_21ep"
echo "Overnight severity-average batch | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (P1=$RUN_P1 P2=$RUN_P2 P3=$RUN_P3 P4=$RUN_P4)"

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
  local logf="logs/overnight_sevavg_${name}.log"
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
    eval "$env_pre $*" > "$logf" 2>&1
    local c=$?
    echo "  [$name] exit=$c"
    if [ $c -ne 0 ]; then
      echo "  WARNING: $name failed, tail of $logf:"
      tail -25 "$logf"
    fi
    return $c
  fi
}

# --- P1: KITTI-C light ---
if [ "$RUN_P1" = "1" ]; then
  _env="EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=light OUT_SUFFIX=sev_light"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p1_kittic_light" "$_env" "$_cmd" || rc=1
fi

# --- P2: KITTI-C moderate ---
if [ "$RUN_P2" = "1" ]; then
  _env="EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=moderate OUT_SUFFIX=sev_moderate"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p2_kittic_moderate" "$_env" "$_cmd" || rc=1
fi

# --- P3: NuScenes-C light ---
if [ "$RUN_P3" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 0 --nusc_c_sev light --extractors cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT --out robust_diagnostic/logs/probe_nusc_c_w0source_sev_light.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p3_nuscc_light" "$_env" "$_cmd" || rc=1
fi

# --- P4: NuScenes-C moderate ---
if [ "$RUN_P4" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 0 --nusc_c_sev moderate --extractors cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT --out robust_diagnostic/logs/probe_nusc_c_w0source_sev_moderate.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p4_nuscc_moderate" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full overnight with: bash run_overnight_sevavg.sh $GPU"
  exit 0
fi
echo "=== OVERNIGHT SEV-AVG DONE ==="
echo "P1 KITTI-C light:     robust_diagnostic/logs/al_full_dataset_ep10_custom_sev_light.json"
echo "P2 KITTI-C moderate:  robust_diagnostic/logs/al_full_dataset_ep10_custom_sev_moderate.json"
echo "P3 NuScenes-C light:  robust_diagnostic/logs/probe_nusc_c_w0source_sev_light.json"
echo "P4 NuScenes-C moderate: robust_diagnostic/logs/probe_nusc_c_w0source_sev_moderate.json"
echo "Average the 3 severities (heavy exists) per condition for the GeoID comparison."
exit 0
