#!/bin/bash
# ==============================================================================
# Tier 0 & 1: headroom ceilings and the severity sweep
# ==============================================================================
# Changes vs the first draft:
#
#  1. SEVERITY GUARD. SEVERITY_MAP = {1:light, 2:moderate, 3:heavy, 4:extreme}
#     and the runner does SEVERITY_MAP.get(sev, 'moderate'). So `--severity 5`
#     silently loads the MODERATE data -- easier than severity 3 -- and the sweep
#     would have shown "more severity doesn't help" without ever testing it.
#     Stage 0 now verifies the directories exist before anything runs.
#
#  2. PRIOR ORACLE ADDED. The scheduled `oracle` bounds GATING. It says nothing
#     about the prior, because prior estimation changes predictions, not which
#     points feed the update. The prior oracle is the number that decides whether
#     the pivot is worth starting. Apply prior_oracle_patch.md first.
#
#  3. BOTH SIDES OF THE REGRESSION. The fit gain = 2.88 - 0.097*frozen crosses
#     zero at 29.6. Testing only harder severities tests one side. Severity 1/2
#     give the easy side and make the crossover claim a fit rather than a guess.
#
#  4. --fire_th, --skip_done, --force_stats, multi-seed on the decisive rows.
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
ROOT="logs/tier_tests"
PANEL="fog,wet_ground,snow,motion_blur,beam_missing,crosstalk,incomplete_echo,cross_sensor"
WORKERS="${WORKERS:-4}"
FIRE_TH="${FIRE_TH:-0.05}"     # 0.0 disables the veto entirely and re-dilutes step size
STATS="logs/source_stats_cache.pt"

mkdir -p "$ROOT"
MAIN_LOG="$ROOT/tier_tests_sweep.log"
say () { echo "$@" | tee -a "$MAIN_LOG"; }

run () {  # run <subdir> <ablations> <severity> <seeds>
  local sub="$1" abl="$2" sev="$3" seeds="$4"
  say ""
  say "--- $sub  (ablations=$abl, severity=$sev, seeds=$seeds)  $(date '+%F %H:%M:%S')"
  uv run ablation_kitti-c.py \
    --pretrained_path "$PRETRAINED" \
    --log_dir "$ROOT/$sub" \
    --corruptions "$PANEL" \
    --severity "$sev" \
    --chunked --reset_per_corruption \
    --ablations "$abl" \
    --seeds "$seeds" \
    --fire_th "$FIRE_TH" \
    --num_workers "$WORKERS" \
    --stats_cache "$STATS" \
    --skip_done 2>&1 | tee -a "$MAIN_LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && say "!! $sub exited rc=$rc -- rerun to resume via --skip_done"
  return 0
}

say "=========================================================="
say "Tier 0 & 1 suite   started $(date)"
say "fire_th=$FIRE_TH  workers=$WORKERS  threads/proc=$ABLATION_THREADS"
say "=========================================================="

# ---------- STAGE 0: severity directories must actually exist ----------
say ""
say "########## STAGE 0: severity directory guard ##########"
declare -A SEVDIR=( [1]=light [2]=moderate [3]=heavy [4]=extreme )
AVAILABLE=""
for s in 1 2 3 4; do
  d="$KITTIC/fog/${SEVDIR[$s]}"
  if [ -d "$d" ]; then
    say "  severity $s -> ${SEVDIR[$s]}   FOUND"
    AVAILABLE="$AVAILABLE $s"
  else
    say "  severity $s -> ${SEVDIR[$s]}   MISSING ($d)"
  fi
done
say ""
say "  NOTE: severity 5 has no entry in SEVERITY_MAP. Requesting it silently"
say "        falls back to 'moderate'. It is deliberately NOT run below."
say "  usable severities:$AVAILABLE"
if [ -z "$AVAILABLE" ]; then
  say "!! no severity directories found -- check KITTIC=$KITTIC"; exit 1
fi

# ---------- STAGE 1: ceilings at severity 3 (the decisive stage) ----------
# 'frozen' MUST come first: prior_oracle reads the chunk GT prior from its
# confusion matrix. 3 seeds because these two numbers gate the whole pivot.
say ""
say "########## STAGE 1: ceilings @ sev 3 ##########"
say "  frozen | prior_oracle (bounds the PRIOR pivot) | full_method | oracle (bounds GATING)"
run tier0_ceiling "frozen,prior_oracle,full_method,oracle" 3 "42,43,44"

say ""
say ">>> READ THIS BEFORE CONTINUING:"
say ">>>   L1(pi_chunk, pi_source) ~ 0  => no prior drift => the pivot is dead,"
say ">>>       tau is a fixed constant, not something to adapt online."
say ">>>   oracle - frozen ~ 0          => no gating scheme can help either."
say ">>> If BOTH are ~0 the honest conclusion is that TTA has no headroom in this"
say ">>> setting, and the next move is a harder setting, not a new mechanism."
grep -E "prior_oracle\]|L1\(pi_chunk" "$ROOT/tier0_ceiling/ablation.log" 2>/dev/null | tail -20 | tee -a "$MAIN_LOG"

# ---------- STAGE 2: severity sweep, BOTH sides of the crossover ----------
# fit: gain = 2.88 - 0.0972 * frozen_mIoU, crossover at frozen = 29.6
# sev 3 sits at mean frozen 33.68, i.e. just above it. Easier severities should
# make the gain MORE negative; harder should flip it positive. Testing only the
# hard side cannot distinguish "the fit is right" from "harder is just better".
say ""
say "########## STAGE 2: severity sweep ##########"
for s in $AVAILABLE; do
  [ "$s" = "3" ] && continue     # already covered by Stage 1
  run "tier1_sev$s" "frozen,full_method" "$s" "42"
done

# ---------- STAGE 3: analysis ----------
say ""
say "########## STAGE 3: analysis ##########"
for d in "$ROOT"/tier*/; do
  [ -f "$d/records.json" ] || continue
  say ""
  say "################### $(basename "$d") ###################"
  uv run analyze_ablations.py --records "$d/records.json" --pct 2>&1 | tee -a "$MAIN_LOG"
done

say ""
say "########## crossover fit across severities ##########"
uv run python - << 'PYEOF' 2>&1 | tee -a "$MAIN_LOG"
import json, glob, os, statistics as st
pts = []
for f in sorted(glob.glob("logs/tier_tests/tier*/records.json")):
    recs = json.load(open(f))
    fr = [r for r in recs if r["ablation"] == "frozen"]
    fu = [r for r in recs if r["ablation"] == "full_method"]
    if not fr or not fu:
        continue
    sev = fr[0]["severity"]
    mf = st.mean(r["final_miou"] for r in fr)
    mu = st.mean(r["final_miou"] for r in fu)
    pts.append((sev, mf * 100, (mu - mf) * 100))
if len(pts) < 2:
    print("need at least two severities to fit"); raise SystemExit
print(f"\n  {'sev':>4}{'mean frozen':>14}{'mean gain':>12}")
for s, f, g in sorted(pts):
    print(f"  {s:>4}{f:>14.2f}{g:>+12.2f}")
x = [p[1] for p in pts]; y = [p[2] for p in pts]
mx, my = st.mean(x), st.mean(y)
den = sum((a-mx)**2 for a in x)
if den > 0:
    b = sum((a-mx)*(c-my) for a, c in zip(x, y)) / den
    a0 = my - b*mx
    print(f"\n  fit: gain = {a0:+.2f} {b:+.4f} * frozen")
    if b != 0:
        print(f"  crossover at frozen mIoU = {-a0/b:.1f}   (sev-3-only estimate was 29.6)")
    print("\n  If the crossover reproduces and the sign flips on the hard side, the")
    print("  headroom explanation holds and severity is the honest operating regime.")
    print("  If it does not, headroom is NOT the explanation and the mechanism is.")
PYEOF

say ""
say "=========================================================="
say "completed $(date)"
say "records: $ROOT/tier*/records.json"
say "=========================================================="
