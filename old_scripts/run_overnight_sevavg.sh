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
#   P0 train NuScenes HyperLiDAR (plain Trainer)  : ~2-4h, ONLY if checkpoint missing
#      AND TRAIN_HYPER=1 (default 0 -- likely exceeds the ~8h window)
#   P1 KITTI-C light   : cov_ep10 + hyper, all 8 conds, severity=light
#   P2 KITTI-C moderate: cov_ep10 + hyper, all 8 conds, severity=moderate
#   P3 NuScenes-C light:  cov_nusc + dgl_nusc + hyper_nusc, all 8 conds, severity=light
#   P4 NuScenes-C moderate: cov_nusc + dgl_nusc + hyper_nusc, all 8 conds, severity=moderate
#
# HyperLiDAR = the STANDARD supervised pretrained feature extractor: the plain
# ResNet-34 built by modules/trainer.py's Trainer() (the base that unsup_main.py
# wraps with hdc_sub.pth). It is NOT supcon_vib. KITTI endpoint is the standard
# checkpoint logs/kitti_pretrain/SENet, loaded with method='baseline' (plain
# arch: no logvar_head / input_in / DGLSS). NuScenes endpoint is
# logs/nusc_pretrain/SENet if it exists; else P0 (opt-in).
#
# Each phase writes a FRESH output file (no stale-info / skip collisions).
#
# Usage:
#   DRY_RUN=1 bash run_overnight_sevavg.sh 2
#   SMOKE=1   bash run_overnight_sevavg.sh 2
#   bash run_overnight_sevavg.sh 2
#   RUN_P0=0 RUN_P1=0 RUN_P2=0 bash run_overnight_sevavg.sh 2   # NuScenes only
#   TRAIN_HYPER=0 bash run_overnight_sevavg.sh 2   # skip P0 even if checkpoint missing

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_P4="${RUN_P4:-1}"
TRAIN_HYPER="${TRAIN_HYPER:-0}"   # 0 = never train (default; ~2-4h, likely exceeds window)
COV_CKPT="robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
COV_NUSC_CKPT="robust_diagnostic/logs/nusc_covshift_21ep"
DGL_NUSC_CKPT="robust_diagnostic/logs/nusc_dglsspp_21ep"
# HyperLiDAR = the STANDARD supervised pretrained feature extractor (plain
# ResNet-34 from modules/trainer.py's Trainer(), the base that unsup_main.py's
# hdc_sub.pth wraps). It is NOT supcon_vib. Loaded with method='baseline'
# (plain arch, no logvar_head / input_in) from logs/kitti_pretrain/SENet.
# NuScenes plain checkpoint may exist at logs/nusc_pretrain/SENet; if missing,
# P0 training is optional and likely exceeds the ~8h window (TRAIN_HYPER=0).
HYPER_CKPT="logs/kitti_pretrain"
HYPER_NUSC_CKPT="logs/nusc_pretrain"
HYPER_METHOD="baseline"
echo "Overnight severity-average batch | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (P0=$RUN_P0 P1=$RUN_P1 P2=$RUN_P2 P3=$RUN_P3 P4=$RUN_P4)"

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
    # CRITICAL: clear any inherited smoke env (MAX_FRAMES/CONDS exported in the
    # shell from a prior SMOKE=1 run) so the FULL run is not capped at 30 frames
    # or restricted to fog. This leaked before (output had _f30 and n_val 2.3M).
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

# --- P0 (optional): train NuScenes HyperLiDAR (plain supervised Trainer) ---
# Only runs if TRAIN_HYPER=1 AND the checkpoint is missing. Default 0 because
# 21ep / 100% is ~2-4h and likely exceeds the ~8h window alongside the eval
# phases. Uses train_covshift_nuscenes.py with --method baseline (plain arch).
if [ "$RUN_P0" = "1" ] && [ "$TRAIN_HYPER" = "1" ]; then
  if [ -f "$HYPER_NUSC_CKPT/SENet" ]; then
    echo "  [P0] NuScenes HyperLiDAR checkpoint exists at $HYPER_NUSC_CKPT/SENet -- skipping train"
  else
    _env="CUDA_VISIBLE_DEVICES=$GPU"
    _cmd="uv run python robust_diagnostic/train_covshift_nuscenes.py --method baseline --epochs 21 --cutoff 1.0 --log_dir $HYPER_NUSC_CKPT"
    run_phase "p0_train_nusc_hyper" "$_env" "$_cmd" || rc=1
  fi
fi

# --- P1: KITTI-C light ---
if [ "$RUN_P1" = "1" ]; then
  _env="EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT,hyper:${HYPER_METHOD}:$HYPER_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=light"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p1_kittic_light" "$_env" "$_cmd" || rc=1
fi

# --- P2: KITTI-C moderate ---
if [ "$RUN_P2" = "1" ]; then
  _env="EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT,hyper:${HYPER_METHOD}:$HYPER_CKPT\" NUSC=0 BAL=1 KITTIC_SEV=moderate"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p2_kittic_moderate" "$_env" "$_cmd" || rc=1
fi

# --- P3: NuScenes-C light ---
if [ "$RUN_P3" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _extractors="cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT"
  if [ -f "$HYPER_NUSC_CKPT/SENet" ]; then
    _extractors="$_extractors,hyper_nusc:${HYPER_METHOD}:$HYPER_NUSC_CKPT"
  else
    echo "  [P3] NuScenes HyperLiDAR checkpoint missing at $HYPER_NUSC_CKPT/SENet -- skipping"
  fi
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 0 --nusc_c_sev light --extractors $_extractors --out robust_diagnostic/logs/probe_nusc_c_w0source_sev_light.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p3_nuscc_light" "$_env" "$_cmd" || rc=1
fi

# --- P4: NuScenes-C moderate ---
if [ "$RUN_P4" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _extractors="cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT"
  if [ -f "$HYPER_NUSC_CKPT/SENet" ]; then
    _extractors="$_extractors,hyper_nusc:${HYPER_METHOD}:$HYPER_NUSC_CKPT"
  else
    echo "  [P4] NuScenes HyperLiDAR checkpoint missing at $HYPER_NUSC_CKPT/SENet -- skipping"
  fi
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 0 --nusc_c_sev moderate --extractors $_extractors --out robust_diagnostic/logs/probe_nusc_c_w0source_sev_moderate.json"
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
echo "P1 KITTI-C light:     robust_diagnostic/logs/al_full_dataset_ep10_custom_light.json"
echo "P2 KITTI-C moderate:  robust_diagnostic/logs/al_full_dataset_ep10_custom_moderate.json"
echo "P3 NuScenes-C light:  robust_diagnostic/logs/probe_nusc_c_w0source_sev_light.json"
echo "P4 NuScenes-C moderate: robust_diagnostic/logs/probe_nusc_c_w0source_sev_moderate.json"
echo "HyperLiDAR endpoints (method=$HYPER_METHOD):"
echo "  KITTI: $HYPER_CKPT (added to P1/P2)"
echo "  NuScenes: $HYPER_NUSC_CKPT (added to P3/P4 only if SENet exists)"
echo "Average the 3 severities (heavy exists) per condition for the GeoID comparison."
exit 0
