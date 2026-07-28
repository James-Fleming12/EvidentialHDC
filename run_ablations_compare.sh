#!/bin/bash
# ==============================================================================
# Corrected Ablation Comparison Suite  (resource-safe, resumable)
# ==============================================================================
# CHANGES vs the first version, after the tmux crash:
#
#  1. THREAD CAPS. Nothing previously limited CPU threads. PyTorch defaults
#     intra-op threads to the core count and every DataLoader worker inherits
#     it, so total runnable threads was ~(1 + workers) x cores -- on a 64-core
#     box with 12 workers that is ~830 threads, 13x oversubscription. That is
#     the classic way to drive load average into the hundreds and make the
#     machine unresponsive, which kills the tmux SERVER, not just your pane.
#
#  2. FEWER DATALOADER WORKERS. ARCH['train']['workers'] is tuned for training
#     with large batches. At batch_size=1 inference it buys little and costs a
#     process pool per pass (>10k pool spawns across a full night).
#
#  3. SOURCE-STAT CACHE. Each stage is a separate process and each was redoing
#     populate_source_statistics (550 frames) before any ablation started.
#
#  4. RESUME. --skip_done skips (seed, ablation, corruption) triples already in
#     records.json, so a crash costs you one cell, not the whole stage.
#
#  5. SURVIVES DISCONNECT. Per-stage `tee -a` instead of one pipeline-wide tee,
#     and it is meant to be launched detached, so a dying terminal cannot
#     SIGPIPE the run.
#
# LAUNCH LIKE THIS (not plain `bash run_...sh`):
#
#     mkdir -p logs/ablation_compare
#     setsid nohup bash run_ablations_compare.sh \
#         > logs/ablation_compare/console.log 2>&1 < /dev/null &
#     disown
#     tail -f logs/ablation_compare/console.log
#
# Then tmux dying is irrelevant. To resume after any interruption, rerun the
# exact same command -- every stage passes --skip_done.
# ==============================================================================

set -uo pipefail

# ---- resource caps (item 1) --------------------------------------------------
# Peak CPU ~ (1 + WORKERS) x ABLATION_THREADS. Keep that under your core count.
# Defaults are deliberately conservative: 5 x 2 = 10 threads.
export ABLATION_THREADS="${ABLATION_THREADS:-2}"
export OMP_NUM_THREADS="$ABLATION_THREADS"
export MKL_NUM_THREADS="$ABLATION_THREADS"
export OPENBLAS_NUM_THREADS="$ABLATION_THREADS"
export NUMEXPR_NUM_THREADS="$ABLATION_THREADS"
export VECLIB_MAXIMUM_THREADS="$ABLATION_THREADS"
WORKERS="${WORKERS:-4}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

PRETRAINED="logs/kitti_pretrain/hdc_sub.pth"
ROOT="logs/ablation_compare"
PANEL="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3
PROTO="--chunked --reset_per_corruption"
FIRE_TH=0.0
STATS_CACHE="logs/source_stats_cache.pt"

# Set from Stage 1 output before Stage 4 is meaningful.
GAP_LO=0.35
GAP_HI=0.75

mkdir -p "$ROOT"
MAIN_LOG="$ROOT/run_ablations_compare.log"

say () { echo "$@" | tee -a "$MAIN_LOG"; }

run () {  # run <subdir> <ablation-set> <seeds> [extra args...]
  local sub="$1"; local abl="$2"; local seeds="$3"; shift 3
  say ""
  say "----------------------------------------------------------"
  say ">>> $sub  (set=$abl, seeds=$seeds)   $(date '+%F %H:%M:%S')"
  say "----------------------------------------------------------"
  uv run ablation_kitti-c.py \
    --pretrained_path "$PRETRAINED" \
    --ablations "$abl" \
    --corruptions "$PANEL" \
    --severity $SEV $PROTO \
    --seeds "$seeds" \
    --fire_th $FIRE_TH \
    --num_workers $WORKERS \
    --stats_cache "$STATS_CACHE" \
    --skip_done \
    --gap_lo $GAP_LO --gap_hi $GAP_HI \
    --log_dir "$ROOT/$sub" "$@" 2>&1 | tee -a "$MAIN_LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    say "!! $sub exited rc=$rc -- continuing (rerun to resume via --skip_done)"
  fi
  return 0
}

say "=========================================================="
say "Corrected Ablation Comparison Suite"
say "started        : $(date)"
say "host / cores   : $(hostname) / $(nproc 2>/dev/null || echo '?')"
say "threads/proc   : $ABLATION_THREADS"
say "dl workers     : $WORKERS   (peak CPU ~ $(( (1+WORKERS) * ABLATION_THREADS )) threads)"
say "protocol       : $PROTO"
say "fire_th        : $FIRE_TH"
say "=========================================================="

# ---------- Stage 0: dry run + live-path check ----------
say ""
say "########## STAGE 0: dry run / live-path check ##########"
uv run ablation_kitti-c.py \
  --pretrained_path "$PRETRAINED" \
  --ablations core --corruptions snow --severity $SEV $PROTO \
  --seeds 42 --dry_run --fire_th $FIRE_TH \
  --num_workers $WORKERS \
  --log_dir "$ROOT/stage0_dryrun" 2>&1 | tee -a "$MAIN_LOG"

say ""
if grep -q "NEVER invoked" "$ROOT/stage0_dryrun/ablation.log" 2>/dev/null; then
  say ">>> !!! HDC_utils gating is INERT -- evaluate_and_adapt uses inline gating."
  say ">>> !!! Every fix in Section 8.3-8.6 is inactive. Stop and wire it in."
else
  say ">>> OK: HDC_utils gating is live."
fi

# ---------- Stage 1: calibrate the domain-gap floor ----------
say ""
say "########## STAGE 1: gap calibration (clean source) ##########"
uv run ablation_kitti-c.py \
  --pretrained_path "$PRETRAINED" \
  --calibrate_gap --severity $SEV --fire_th $FIRE_TH \
  --num_workers $WORKERS --stats_cache "$STATS_CACHE" \
  --log_dir "$ROOT/stage1_gapcalib" 2>&1 | tee -a "$MAIN_LOG"
say ""
say ">>> Read 'set --gap_lo X --gap_hi Y' above, put them at the top of this"
say ">>> script, then rerun -- Stages 2-3 skip via --skip_done and only the"
say ">>> Stage 4 gain configs recompute."

# ---------- Stage 2: core contrast, 3 seeds -> noise floor ----------
say ""
say "########## STAGE 2: core, 3 seeds ##########"
run stage2_core core "42,43,44"

# ---------- Stage 3: attribution ----------
say ""
say "########## STAGE 3: leave-one-out + add-one-in ##########"
run stage3_loo loo "42"
run stage3_aoi aoi "42"

# ---------- Stage 4: gates, presets, gain, ceiling ----------
say ""
say "########## STAGE 4: gate zoo / presets / gain / ceiling ##########"
run stage4_preset  preset  "42"
run stage4_zoo     zoo     "42"
run stage4_gain    gain    "42"
run stage4_ceiling ceiling "42"

# ---------- Stage 5: analysis ----------
say ""
say "########## STAGE 5: analysis ##########"
for d in stage2_core stage3_loo stage3_aoi stage4_preset stage4_zoo stage4_gain stage4_ceiling; do
  if [ -f "$ROOT/$d/records.json" ]; then
    say ""
    say "################### $d ###################"
    uv run analyze_ablations.py --records "$ROOT/$d/records.json" --pct 2>&1 | tee -a "$MAIN_LOG"
  fi
done

say ""
say "=========================================================="
say "completed $(date)"
say "records : $ROOT/stage*/records.json"
say "logs    : $MAIN_LOG  and  $ROOT/stage*/ablation.log"
say "=========================================================="