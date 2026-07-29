#!/bin/bash
# ==============================================================================
# run_diagnostic_tests.sh
# 
# Executes the prioritized diagnostic plan (D0 - D5) from docs/fourth_iteration.md
# to determine if geometric TTA is fundamentally broken or just wired incorrectly.
# ==============================================================================

set -uo pipefail

export ABLATION_THREADS="${ABLATION_THREADS:-2}"
export OMP_NUM_THREADS="$ABLATION_THREADS"
export MKL_NUM_THREADS="$ABLATION_THREADS"
export OPENBLAS_NUM_THREADS="$ABLATION_THREADS"
export NUMEXPR_NUM_THREADS="$ABLATION_THREADS"
export VECLIB_MAXIMUM_THREADS="$ABLATION_THREADS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PRETRAINED="logs/kitti_pretrain/hdc_sub.pth"
KITTIC="${KITTIC:-/mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C/SemanticKITTI-C}"
ROOT="logs/diagnostic_tests"
PANEL="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
WORKERS="${WORKERS:-4}"
FIRE_TH="${FIRE_TH:-0.05}"
STATS="logs/source_stats_cache.pt"

mkdir -p "$ROOT"
MAIN_LOG="$ROOT/diagnostic_tests.log"
say () { echo "$@" | tee -a "$MAIN_LOG"; }

run () {
  local sub="$1" abl="$2" sev="$3" seeds="$4" corrs="${5:-$PANEL}"
  say ""
  say "--- $sub  (ablations=$abl, severity=$sev, seeds=$seeds)  $(date '+%F %H:%M:%S')"
  uv run ablation_kitti-c.py \
    --pretrained_path "$PRETRAINED" \
    --log_dir "$ROOT/$sub" \
    --corruptions "$corrs" \
    --severity "$sev" \
    --chunked --reset_per_corruption \
    --ablations "$abl" \
    --seeds "$seeds" \
    --fire_th "$FIRE_TH" \
    --num_workers "$WORKERS" \
    --stats_cache "$STATS" \
    --skip_done 2>&1 | tee -a "$MAIN_LOG"
}

say "=========================================================="
say "Starting D0-D5 Diagnostic Tests"
say "=========================================================="

# 1. D0 (The Fix) & D5 (The Ceilings)
# This includes the critical 'full_method_d0b' which tests the consistent tau gating fix.
# It also bounds the problem with 'oracle' (gate ceiling) and 'prior_oracle' (prior ceiling).
run "diag_ceilings_and_fix" "frozen,prior_oracle,oracle,full_method,full_method_d0b,full_method_d0b_veto" 3 "42,43,44"

# 2. PRIOR REMOVAL: does the tau prior on pseudo-labels help or hurt adaptation?
run "diag_prior_removal" "prior" 3 "42,43,44"

# 3. RECOVERY: reintroduce the prelim annealing schedule deliberately, on the D0b base
run "diag_recovery" "recover" 3 "42,43,44"

# 4. D3 (Component Interference / Interaction)
# Ladder of components to check if they impede each other.
run "diag_components" "aoi_1_tau,aoi_2_gate,aoi_3_bm,aoi_4_ic4,no_dual_gating" 3 "42"

# 3. D4 (Invariants Check)
# Run a single corruption to ensure determinism matches the 8-panel run.
# If frozen mIoU for fog here doesn't perfectly match fog in the 8-panel run above, the data chunking is non-deterministic.
run "diag_invariants_single_corr" "frozen,full_method" 3 "42" "fog"

say "=========================================================="
say "completed $(date)"
say "logs saved to $ROOT"
say "=========================================================="
