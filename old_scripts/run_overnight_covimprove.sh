#!/usr/bin/env bash
# run_overnight_covimprove.sh: overnight batch for the cov-shift improvement paths
# (docs/cov_shift/cov_full_scale.md) + the discrepancy re-runs.
#
# Modes:
#   DRY_RUN=1  print each command without running (quick syntax/path check).
#   SMOKE=1    actually run each phase with tiny MAX_FRAMES + fog only, and
#              FAIL on any phase that errors -- derisk every script before the
#              full overnight slot.
#   (default)  full overnight run.
#
# Phases (each independently logged; a failure in one does not stop the rest):
#   P1 DISCREPANCY RERUN: authoritative DGLSS++ KITTI-C ceilings
#      (the corrected-run JSON was never pulled; mechanism-probe ceilings are
#       unreliable due to a ~0.5% n_val mismatch). NUSC=0 (KITTI-C only).
#   P2 IMPROVEMENT 4: full-scale code-2000 verification
#      (does the tta Iteration-2 "peak at 2000" hold at full scale? cov_ep10).
#   P3 IMPROVEMENT 1: D7 gate test, CORRECTED -- fresh model built with
#      input_in=False (the mechanism probe's in-place toggle was inert).
#   P4 D9-CORRECTED NuScenes-C: zero-shot with in-domain nuScenes-clean W0,
#      ceiling via the authoritative eval_target_condition (one consistent run).
#   P5 IMPROVEMENT 4 (NuScenes-C leg): same as P4 but PROJ_DIM=2000 -- does the
#      code-2000 peak transfer to NuScenes-C? (P4 and P5 write separate out files.)
#
# Usage:
#   DRY_RUN=1 bash run_overnight_covimprove.sh            # print only
#   SMOKE=1   bash run_overnight_covimprove.sh 2          # error-check each phase
#   bash run_overnight_covimprove.sh 2                    # full overnight
#   RUN_P1=0 RUN_P2=0 bash run_overnight_covimprove.sh 2  # subset

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_P4="${RUN_P4:-1}"
RUN_P5="${RUN_P5:-1}"
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
COV_CKPT="robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
COV_NUSC_CKPT="robust_diagnostic/logs/nusc_covshift_21ep"
DGL_NUSC_CKPT="robust_diagnostic/logs/nusc_dglsspp_21ep"
echo "Overnight cov-improve batch | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (P1=$RUN_P1 P2=$RUN_P2 P3=$RUN_P3 P4=$RUN_P4 P5=$RUN_P5)"

# Smoke caps: tiny frame count + one condition so each phase completes in ~1-3 min.
SM_FRAMES="${SM_FRAMES:-30}"
SM_CONDS="fog"

# wrapper: echo the command, optionally execute, return failure if it errors.
#   run_phase <name> <env-prefix> <command...>
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
  local logf="logs/overnight_${name}.log"
  if [ "$SMOKE" = "1" ]; then
    # smoke: run in foreground, capture output, fail fast on error
    echo "  [SMOKE] executing (max_frames=${SM_FRAMES}, conds=${SM_CONDS})..."
    if eval "$env_pre $*" > "$logf" 2>&1; then
      echo "  [SMOKE] $name OK"
    else
      echo "  [SMOKE] $name FAILED -- see $logf (tail below):"
      tail -25 "$logf"
      return 1
    fi
  else
    echo "  [RUN] full phase, logging to $logf"
    eval "$env_pre $*" > "$logf" 2>&1
    local rc=$?
    echo "  [$name] exit=$rc"
    if [ $rc -ne 0 ]; then
      echo "  WARNING: $name failed, tail of $logf:"
      tail -25 "$logf"
    fi
  fi
}

rc=0

# --- P1: authoritative DGLSS++ KITTI-C (discrepancy rerun) ---
if [ "$RUN_P1" = "1" ]; then
  _env="EXTRACTORS=\"dglsspp:supcon_vib_dglsspp:$DGLSSPP_CKPT\" NUSC=0 PROJ_DIM=10000 OUT_SUFFIX=dglsspp"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p1_dglsspp_kittic" "$_env" "$_cmd" || rc=1
fi

# --- P2: code-2000 full-scale verification (cov_ep10) ---
if [ "$RUN_P2" = "1" ]; then
  _env="EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT\" NUSC=0 PROJ_DIM=2000 OUT_SUFFIX=dim2000"
  _cmd="CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU"
  if [ "$SMOKE" = "1" ]; then _env="$_env MAX_FRAMES=$SM_FRAMES CONDS=$SM_CONDS"; fi
  run_phase "p2_codim2000" "$_env" "$_cmd" || rc=1
fi

# --- P3: D7 gate test, corrected (fresh input_in=False build) ---
if [ "$RUN_P3" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/probe_d7_gate_diag.py --skip_existing 1 --out robust_diagnostic/logs/probe_d7_gate_ep10.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p3_d7gate" "$_env" "$_cmd" || rc=1
fi

# --- P4: corrected NuScenes-C zero-shot (in-domain W0) ---
if [ "$RUN_P4" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 1 --extractors cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT --out robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p4_w0source" "$_env" "$_cmd" || rc=1
fi

# --- P5: NuScenes-C code-2000 verification (cov_nusc + dgl_nusc) ---
# Does the tta Iteration-2 "peak at 2000" transfer to NuScenes-C? Uses the same
# full harness (in-domain W0 via P4's mechanism, PROJ_DIM=2000). Note: P4 and P5
# write different out files (P4 = 10000-d in-domain W0, P5 = 2000-d), so no clash.
if [ "$RUN_P5" = "1" ]; then
  _env="CUDA_VISIBLE_DEVICES=$GPU"
  _cmd="uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py --skip_existing 1 --proj_dim 2000 --extractors cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT --out robust_diagnostic/logs/probe_nusc_c_w0source_dim2000.json"
  if [ "$SMOKE" = "1" ]; then _cmd="$_cmd --max_frames $SM_FRAMES"; fi
  run_phase "p5_nuscc_dim2000" "$_env" "$_cmd" || rc=1
fi

echo ""
if [ "$SMOKE" = "1" ] && [ $rc -ne 0 ]; then
  echo "=== SMOKE RESULT: FAILURES DETECTED (fix before the full run) ==="
  exit 1
fi
if [ "$SMOKE" = "1" ]; then
  echo "=== SMOKE RESULT: ALL PHASES OK ==="
  echo "Launch the full overnight with: bash run_overnight_covimprove.sh $GPU"
  exit 0
fi
echo "=== OVERNIGHT COV-IMPROVE DONE ==="
echo "P1: robust_diagnostic/logs/al_full_dataset_ep10_custom_dglsspp.json (authoritative DGLSS++ KITTI-C)"
echo "P2: robust_diagnostic/logs/al_full_dataset_ep10_custom_dim2000.json (code-2000, cov_ep10)"
echo "P3: robust_diagnostic/logs/probe_d7_gate_ep10.json (input_in=False ceilings)"
echo "P4: robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json (in-domain W0 zero-shot)"
exit 0
