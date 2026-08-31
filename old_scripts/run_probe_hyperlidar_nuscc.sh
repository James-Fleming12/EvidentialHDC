#!/usr/bin/env bash
# run_probe_hyperlidar_nuscc.sh: compute R1 (prototype) and R4 (linear probe)
# zero-shot + ceiling for the ORIGINAL HyperLiDAR feature extractor (the plain
# supervised `baseline` = CENET-style SENet) on NuScenes-C.
#
# The overnight run trained the NuScenes HyperLiDAR extractor to
# logs/nusc_pretrain (21 ep / 100%, method `baseline`). This evaluates it on
# nuScenes-C with the in-domain nuScenes-clean W0 (the corrected zero-shot
# protocol, per D9), reporting per condition:
#   R4 linear : linear_frozen (zero-shot) / linear_ceiling (labeled bound)
#   R1 proto  : proto_frozen (zero-shot)  / proto_ceiling (labeled bound)
# Also reports the contaminated KITTI-clean-W0 frozen for reference.
#
# Modes:
#   DRY_RUN=1  print the command without running
#   SMOKE=1    run at tiny MAX_FRAMES + fog only, fail on error
#   (default)  full run (all 8 conditions, heavy severity)
#
# Usage:
#   DRY_RUN=1 bash run_probe_hyperlidar_nuscc.sh 2
#   SMOKE=1   bash run_probe_hyperlidar_nuscc.sh 2
#   bash run_probe_hyperlidar_nuscc.sh 2
#   NUSC_C_SEV=moderate bash run_probe_hyperlidar_nuscc.sh 2   # other severity
#
# Output: robust_diagnostic/logs/probe_hyperlidar_nuscc_heavy.json
#   extractors.hyper_nusc.conds[<cond>] = { linear_frozen/linear_ceiling,
#     proto_frozen/proto_ceiling, ... }

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
NUSC_C_SEV="${NUSC_C_SEV:-heavy}"
SM_FRAMES="${SM_FRAMES:-10}"
HYPER_NUSC_CKPT="${HYPER_NUSC_CKPT:-logs/nusc_pretrain}"
OUT="robust_diagnostic/logs/probe_hyperlidar_nuscc_${NUSC_C_SEV}.json"
echo "HyperLiDAR NuScenes-C probe | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE (sev=$NUSC_C_SEV, ckpt=$HYPER_NUSC_CKPT)"

if [ ! -f "$HYPER_NUSC_CKPT/SENet" ]; then
  echo "ERROR: $HYPER_NUSC_CKPT/SENet not found -- did the overnight P1 finish?"
  exit 1
fi

CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_nusc_c_w0source_diag.py \
  --nusc_c_sev $NUSC_C_SEV \
  --extractors hyper_nusc:baseline:$HYPER_NUSC_CKPT \
  --out $OUT"
if [ "$SMOKE" = "1" ]; then CMD="$CMD --max_frames $SM_FRAMES"; fi
echo "CMD: $CMD"

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] not executed"
  exit 0
fi

eval "$CMD" 2>&1 | tee "logs/probe_hyperlidar_nuscc_${NUSC_C_SEV}.log"
RC=${PIPESTATUS[0]}

if [ $RC -eq 0 ]; then
  echo "=== HYPERLIDAR NUSC-C OK ==="
  echo "  R1 proto frozen/ceiling + R4 linear frozen/ceiling per condition: $OUT"
else
  echo "=== HYPERLIDAR NUSC-C FAILED (exit $RC) ==="
  exit $RC
fi
