#!/bin/bash
# ==============================================================================
# Corrected Ablation Comparison Suite
# ==============================================================================
# Staged so partial results are usable if you stop it early. Each stage writes
# its own records.json, and stage 5 analyses whatever exists.
#
# Stage 0  dry run + live-path check      ~5 min
# Stage 1  gap calibration on clean src   ~15 min
# Stage 2  core, 3 seeds  -> noise floor  ~2 h
# Stage 3  leave-one-out + add-one-in     ~6 h
# Stage 4  gate zoo + presets + gain      ~6 h
# Stage 5  analysis                       ~1 min
#
# IMPORTANT: after Stage 0, check the log for
#     "HDC_utils gating was NEVER invoked"
# If that appears, unsup_kitti-c.py is still using its own inline gate block and
# every fix in HDC_utils.py is inert. Fix that before trusting Stages 2-4.
# ==============================================================================

set -uo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3

PRETRAINED="logs/kitti_pretrain/hdc_sub.pth"
ROOT="logs/ablation_compare"
PANEL="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
SEV=3
PROTO="--chunked --reset_per_corruption"    # identical across every stage
FIRE_TH=0.0                                 # see Section 8.3 before changing

# Set these from Stage 1 output before Stage 4 matters. Defaults are guesses.
GAP_LO=0.35
GAP_HI=0.75

mkdir -p "$ROOT"
LOG_FILE="$ROOT/run_ablations_compare.log"

run () {  # run <subdir> <ablation-set> <seeds> [extra args...]
  local sub="$1"; local abl="$2"; local seeds="$3"; shift 3
  echo ""
  echo "----------------------------------------------------------"
  echo ">>> $sub  (set=$abl, seeds=$seeds)   $(date +%H:%M:%S)"
  echo "----------------------------------------------------------"
  uv run ablation_kitti-c.py \
    --pretrained_path "$PRETRAINED" \
    --ablations "$abl" \
    --corruptions "$PANEL" \
    --severity $SEV $PROTO \
    --seeds "$seeds" \
    --fire_th $FIRE_TH \
    --gap_lo $GAP_LO --gap_hi $GAP_HI \
    --log_dir "$ROOT/$sub" "$@" \
    || echo "!! $sub FAILED -- continuing to next stage"
}

{
echo "=========================================================="
echo "Corrected Ablation Comparison Suite"
echo "started        : $(date)"
echo "protocol       : $PROTO"
echo "severity       : $SEV"
echo "fire_th        : $FIRE_TH"
echo "=========================================================="

# ---------- Stage 0: dry run + live-path check ----------
echo ""
echo "########## STAGE 0: dry run / live-path check ##########"
uv run ablation_kitti-c.py \
  --pretrained_path "$PRETRAINED" \
  --ablations core --corruptions snow --severity $SEV $PROTO \
  --seeds 42 --dry_run --fire_th $FIRE_TH \
  --log_dir "$ROOT/stage0_dryrun"

echo ""
echo ">>> CHECK NOW: grep for 'NEVER invoked' in $ROOT/stage0_dryrun/ablation.log"
grep -n "NEVER invoked" "$ROOT/stage0_dryrun/ablation.log" \
  && echo ">>> !!! HDC_utils gating is INERT. Stop and wire it in. !!!" \
  || echo ">>> OK: HDC_utils gating is live."

# ---------- Stage 1: calibrate the domain-gap floor ----------
echo ""
echo "########## STAGE 1: gap calibration (clean source) ##########"
uv run ablation_kitti-c.py \
  --pretrained_path "$PRETRAINED" \
  --calibrate_gap --severity $SEV --fire_th $FIRE_TH \
  --log_dir "$ROOT/stage1_gapcalib"
echo ""
echo ">>> Read 'set --gap_lo X --gap_hi Y' from the log above and put them at the"
echo ">>> top of this script, then rerun Stage 4 if you want calibrated gain control."

# ---------- Stage 2: core contrast, 3 seeds -> noise floor ----------
echo ""
echo "########## STAGE 2: core, 3 seeds ##########"
run stage2_core core "42,43,44"

# ---------- Stage 3: attribution ----------
echo ""
echo "########## STAGE 3: leave-one-out + add-one-in ##########"
run stage3_loo loo "42"
run stage3_aoi aoi "42"

# ---------- Stage 4: gates, presets, gain, ceiling ----------
echo ""
echo "########## STAGE 4: gate zoo / presets / gain / ceiling ##########"
run stage4_preset preset "42"
run stage4_zoo    zoo    "42"
run stage4_gain   gain   "42"
run stage4_ceiling ceiling "42"

# ---------- Stage 5: analysis ----------
echo ""
echo "########## STAGE 5: analysis ##########"
for d in stage2_core stage3_loo stage3_aoi stage4_preset stage4_zoo stage4_gain stage4_ceiling; do
  if [ -f "$ROOT/$d/records.json" ]; then
    echo ""
    echo "################### $d ###################"
    uv run analyze_ablations.py --records "$ROOT/$d/records.json" --pct \
      || echo "!! analysis failed for $d"
  fi
done

echo ""
echo "=========================================================="
echo "completed $(date)"
echo "records: $ROOT/stage*/records.json"
echo "=========================================================="

} 2>&1 | tee "$LOG_FILE"