#!/usr/bin/env bash
# 10h medium run of the dircons decoupling variant (Iteration-17 winner):
#   supcon_vib_dglsspp_corsupcon_residual_128_128_dircons
# inv 128 + corr = inv + dz (residual), L_res=0.05, L_dir=0.1 (EMA displacement
# direction consistency). Trained 21 ep / 100% data. Eval-only afterwards (the
# isotropy_diag harness trains AND evaluates the 8-condition decode at the end).
#
# Run the mid-training monitor IN PARALLEL (bash monitor_dircons.sh 3) to catch a
# reweighting need (dir_w / res_w / lscc_corr) before the 10h is wasted.
#
# Usage:
#   bash run_dircons_medium.sh            # GPU 3

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"
MED_DIR="robust_diagnostic/logs/med_dircons"
echo "Using GPU $GPU, method $METHOD"

CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
  --methods "$METHOD" --epochs 21 --cutoff 1.0 --log_dir "$MED_DIR" \
  2>&1 | tee "logs/dircons_med_train.log"
