#!/usr/bin/env bash
# ~9.5-10h medium run of the dircons decoupling variant (Iteration-17 winner):
#   supcon_vib_dglsspp_corsupcon_residual_128_128_dircons
# inv 128 + corr = inv + dz (residual), L_res=0.05, L_dir=0.1 (EMA displacement
# direction consistency). Defaults to 17 ep / 100% (~30 min/ep) so training lands
# ~9-10h INCLUDING the final 8-condition eval. If the mid-training monitor shows
# convergence isn't done, continue to 21 ep with --resume (no restart needed).
#
# Run the mid-training monitor IN PARALLEL (bash monitor_dircons.sh 3) to catch a
# reweighting need (dir_w / res_w / lscc_corr) before the run is wasted.
#
# Usage:
#   bash run_dircons_medium.sh            # GPU 3, 17 epochs (~9.5h total)
#   bash run_dircons_medium.sh 3 21       # full 21 epochs (~12h)
#   bash run_dircons_medium.sh 3 17 1     # 17 epochs, resume-existing (continue)

set -u
GPU="${1:-3}"
EPOCHS="${2:-17}"
RESUME="${3:-0}"
METHOD="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"
MED_DIR="robust_diagnostic/logs/med_dircons"
echo "Using GPU $GPU, method $METHOD, $EPOCHS epochs${RESUME:+ (resume)}"

RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
  RESUME_FLAG="--resume"
fi

CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$METHOD" --epochs "$EPOCHS" --cutoff 1.0 --log_dir "$MED_DIR" \
  $RESUME_FLAG \
  2>&1 | tee "logs/dircons_med_train.log"
