#!/usr/bin/env bash
# run_probe_linear_prop.sh: what FEATURE-SPACE PROPERTIES does the R4 linear
# classifier need, and which extractor property predicts the recoverable ceiling?
#
# Diagnostic 10 showed geometric invariance does NOT predict the ceiling (hyper
# least invariant on KITTI-C yet highest ceiling; most invariant on NuScenes-C
# yet lowest). This probe measures the properties the LINEAR classifier actually
# uses, in the binarized 10000-d code space where R4 operates:
#   pre_sign_margin_lt05_frac : fraction of codes near the sign-flip boundary
#   fisher_ratio              : between/within scatter (linear separability)
#   within_class_var          : code cluster tightness
#   effrank                   : usable directions (conditioning)
#   margin_sweep              : frozen R4 after zeroing fragile margins
# plus frozen/ceiling for the correlation.
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES (default 10), fog only
#   (default)  full run (200 frames, fog+crosstalk)
#
# Usage:
#   DRY_RUN=1 bash run_probe_linear_prop.sh 2
#   SMOKE=1   bash run_probe_linear_prop.sh 2
#   bash run_probe_linear_prop.sh 2
#
# Output: robust_diagnostic/logs/probe_linear_prop.json

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk}"
SM_FRAMES="${SM_FRAMES:-10}"
EXTRACTORS="${EXTRACTORS:-hyper_kitti:baseline:logs/kitti_pretrain,dgl_kitti:supcon_vib_dglsspp:robust_diagnostic/logs/supcon_vib_dglsspp,cov_kitti:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan,hyper_nusc:baseline:logs/nusc_pretrain,dgl_nusc:supcon_vib_dglsspp:robust_diagnostic/logs/nusc_dglsspp_21ep,cov_nusc:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/nusc_covshift_21ep}"
echo "Linear-properties probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (max_frames=$MAX_FRAMES, conds=$CONDS)"

if [ "$SMOKE" = "1" ]; then
  MAX_FRAMES="$SM_FRAMES"; CONDS="fog"
  echo "  [SMOKE] using max_frames=$MAX_FRAMES, conds=$CONDS"
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python robust_diagnostic/probe_linear_prop_diag.py \
  --max_frames $MAX_FRAMES --conds $CONDS --extractors \"$EXTRACTORS\" \
  --out robust_diagnostic/logs/probe_linear_prop.json"
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD"
RC=$?
echo "  [exit code: $RC]"

if [ $RC -eq 0 ]; then
  echo "=== LINEAR-PROP OK ==="
  echo "  Which property (pre-sign margin / fisher / within-var / effrank) tracks"
  echo "  the ceiling across extractors? That is what to optimize in the extractor."
else
  echo "=== LINEAR-PROP FAILED (exit $RC) ==="
  exit $RC
fi
