#!/usr/bin/env bash
# run_overnight_covimprove.sh: overnight batch for the cov-shift improvement paths
# (docs/cov_shift/cov_full_scale.md) + the discrepancy re-runs.
#
# DRY_RUN=1 prints each command and its GPU without running (derisk the batch
# before committing an overnight slot).
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
#
# Usage:
#   DRY_RUN=1 bash run_overnight_covimprove.sh          # just print, no run
#   bash run_overnight_covimprove.sh 2                  # full overnight on GPU 2
#   RUN_P1=0 RUN_P2=0 bash run_overnight_covimprove.sh 2  # subset

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-1}"
RUN_P3="${RUN_P3:-1}"
RUN_P4="${RUN_P4:-1}"
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
COV_CKPT="robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"
COV_NUSC_CKPT="robust_diagnostic/logs/nusc_covshift_21ep"
DGL_NUSC_CKPT="robust_diagnostic/logs/nusc_dglsspp_21ep"
echo "Overnight cov-improve batch | GPU $GPU | DRY_RUN=$DRY_RUN (P1=$RUN_P1 P2=$RUN_P2 P3=$RUN_P3 P4=$RUN_P4)"
[ "$DRY_RUN" = "1" ] && echo ">>> DRY RUN: printing commands only, nothing executes"

run_cmd() {
  local name="$1"; shift
  echo ""
  echo "==================== $name ===================="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY: $*"
  else
    echo "  RUN: $*"
    eval "$*"
    echo "  [$name] exit=$?"
  fi
}

# --- P1: authoritative DGLSS++ KITTI-C (discrepancy rerun) ---
if [ "$RUN_P1" = "1" ]; then
  run_cmd "P1 dglsspp KITTI-C authoritative" \
    "EXTRACTORS=\"dglsspp:supcon_vib_dglsspp:$DGLSSPP_CKPT\" NUSC=0 PROJ_DIM=10000 OUT_SUFFIX=dglsspp \
     CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU > logs/overnight_p1_dglsspp_kittic.log 2>&1"
fi

# --- P2: code-2000 full-scale verification (cov_ep10) ---
if [ "$RUN_P2" = "1" ]; then
  run_cmd "P2 code-2000 full-scale (cov_ep10)" \
    "EXTRACTORS=\"cov_ep10:supcon_vib_dglsspp_inputin_in_chan:$COV_CKPT\" NUSC=0 PROJ_DIM=2000 OUT_SUFFIX=dim2000 \
     CUDA_VISIBLE_DEVICES=$GPU bash run_al_full_dataset.sh $GPU > logs/overnight_p2_codim2000.log 2>&1"
fi

# --- P3: D7 gate test, corrected (fresh input_in=False build) ---
if [ "$RUN_P3" = "1" ]; then
  run_cmd "P3 D7 gate-off (cov_kitti + cov_nusc)" \
    "CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_d7_gate_diag.py \
     --out robust_diagnostic/logs/probe_d7_gate_ep10.json > logs/overnight_p3_d7gate.log 2>&1"
fi

# --- P4: corrected NuScenes-C zero-shot (in-domain W0) ---
if [ "$RUN_P4" = "1" ]; then
  run_cmd "P4 NuScenes-C in-domain W0 (cov_nusc + dgl_nusc)" \
    "CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py \
     --extractors cov_nusc:supcon_vib_dglsspp_inputin_in_chan:$COV_NUSC_CKPT,dgl_nusc:supcon_vib_dglsspp:$DGL_NUSC_CKPT \
     --out robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json > logs/overnight_p4_w0source.log 2>&1"
fi

echo ""
echo "=== OVERNIGHT COV-IMPROVE DONE ==="
echo "P1: robust_diagnostic/logs/al_full_dataset_ep10_custom_dglsspp.json (authoritative DGLSS++ KITTI-C)"
echo "P2: robust_diagnostic/logs/al_full_dataset_ep10_custom_dim2000.json (code-2000, cov_ep10)"
echo "P3: robust_diagnostic/logs/probe_d7_gate_ep10.json (input_in=False ceilings)"
echo "P4: robust_diagnostic/logs/probe_nusc_c_w0source_ep10.json (in-domain W0 zero-shot)"
