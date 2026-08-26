#!/usr/bin/env bash
# run_probe_rep_robustness.sh: WHY is the plain-supervised HyperLiDAR extractor
# (`baseline`) MORE robust than DGLSS++/cov-shift on KITTI-C but LESS on
# NuScenes-C?
#
# Measures, per (extractor, dataset, condition), with a clean reference:
#   class_shift_clean_to_corr : per-class feature mean cosine (invariance)
#   sep_clean / sep_corr / sep_retention : per-class separability retention
#   bn_mismatch_conv1 : BN running-stat drift (network dynamics)
#   code_var / effrank : code compression / effective rank
#   input_stats : range/remission variance (the D3 input trigger)
#
# Runs kitti extractors on KITTI-C and nusc extractors on NuScenes-C (in-domain),
# so the cross-extractor comparison is apples-to-apples.
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (200 frames, fog+crosstalk+snow)
#
# Usage:
#   DRY_RUN=1 bash run_probe_rep_robustness.sh 2
#   SMOKE=1   bash run_probe_rep_robustness.sh 2
#   bash run_probe_rep_robustness.sh 2
#   CONDS=fog bash run_probe_rep_robustness.sh 2
#
# Output: robust_diagnostic/logs/probe_rep_robustness.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk,snow}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-hyper_kitti:baseline:logs/kitti_pretrain,dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan,hyper_nusc:baseline:logs/nusc_pretrain,dgl_nusc:supcon_vib_dglsspp:robust_diagnostic/logs/nusc_dglsspp_21ep,cov_nusc:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/nusc_covshift_21ep}"
echo "Representation-robustness probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_rep_robustness_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_rep_robustness.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_rep_robustness.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== REP-ROBUSTNESS OK ==="
  echo "  Compare per-dataset: is hyper's class_shift / sep_retention / bn_mismatch"
  echo "  better on KITTI-C but worse on NuScenes-C vs DGLSS++/cov-shift?"
else
  echo "=== REP-ROBUSTNESS FAILED (exit $RC) ==="
  exit $RC
fi
