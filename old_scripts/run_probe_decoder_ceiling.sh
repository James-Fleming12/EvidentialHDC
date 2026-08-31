#!/usr/bin/env bash
# run_probe_decoder_ceiling.sh: is the linear probe on the HDC code the best
# decoder, or is there a HIGHER ceiling? Compares, on the same pool/val:
#   R4 linear (code, reference) | linear (raw 128-d) | kNN1/kNN5 (code) |
#   kNN1 (raw) | RFF kernel ridge (code) | balanced per-class-lam linear
# across conditions. If kNN/RFF >> linear on many conditions, the linear probe
# is NOT the ceiling and a more expressive decoder raises the recoverable gaps.
#
# Usage:
#   bash run_probe_decoder_ceiling.sh 2
#   MAX_FRAMES=100 CONDS=fog,snow bash run_probe_decoder_ceiling.sh 2
#
# Output: robust_diagnostic/logs/probe_decoder_ceiling.json

set -u
set -o pipefail
GPU="${1:-2}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
EXTRACTORS="${EXTRACTORS:-cov_ep10:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan}"
echo "Using GPU $GPU (max_frames=$MAX_FRAMES, conds=$CONDS)"

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_decoder_ceiling_diag.py \
  --max_frames "$MAX_FRAMES" --conds "$CONDS" --extractors "$EXTRACTORS" \
  --out "robust_diagnostic/logs/probe_decoder_ceiling.json" \
  2>&1 | tee "logs/probe_decoder_ceiling.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== DECODER CEILING OK ==="
  echo "  Compare knn1_code / rff_ridge_code vs r4_linear_code per condition:"
  echo "  kNN/RFF >> linear = a more expressive decoder raises the ceiling"
  echo "  kNN/RFF ~ linear  = the linear probe already hits the code's info limit"
else
  echo "=== DECODER CEILING FAILED (exit $RC) ==="
  exit $RC
fi
