#!/usr/bin/env bash
# run_overnight_probe_eff.sh: the longer combined probe-efficiency + AL-gauge run.
#
#   1. run_probe_spectral_trunc.sh   training-side: truncated spectral fit vs
#                                    full eigh/solve vs Nys-warm CG-8/20
#                                    (exact-solve accuracy at CG-class cost?)
#   2. run_probe_decode_quant.sh     inference-side: fp32 vs int8 vs +-1 vs
#                                    low-rank factored W decode (speed + ceiling)
#   3. probe_al_gauge_multi_diag     robust AL-gauge test: the label-free
#                                    "should we do AL?" signals, validated across
#                                    ALL extractors (cov_ep10/21, dglsspp, robust),
#                                    combined score, and a threshold decision test
#                                    (gate routes to wet_ground/fog, skips the rest)
#
# Stages 1-2 run on 2 conditions each (fast, ~15-25 min); stage 3 runs all 8
# conditions x 4 extractors (~1.5-2h). Total ~2-2.5h.
#
# Usage:
#   bash run_overnight_probe_eff.sh 1           # full run on GPU 1
#   DRY_RUN=TRUE bash run_overnight_probe_eff.sh 1   # smoke: 1 cond, 1 extractor
#
# Config via env: GPU, DRY_RUN, CONDS, MAX_FRAMES

set -u
set -o pipefail
GPU="${1:-1}"
DRY_RUN="${DRY_RUN:-FALSE}"

if [ "$DRY_RUN" = "TRUE" ]; then
  echo "=== DRY RUN ==="
  CONDS="${CONDS:-fog}"
  MAX_FRAMES="${MAX_FRAMES:-200}"
  EFF_CONDS="${EFF_CONDS:-fog}"
else
  CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
  MAX_FRAMES="${MAX_FRAMES:-0}"
  EFF_CONDS="${EFF_CONDS:-wet_ground,fog}"
fi
echo "Using GPU $GPU (dry_run=$DRY_RUN, conds=$CONDS, max_frames=$MAX_FRAMES)"

METHOD="supcon_vib_dglsspp_inputin_in_chan"
CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
CKPT21="robust_diagnostic/logs/med_$METHOD/$METHOD"
DGLSSPP_CKPT="robust_diagnostic/logs/supcon_vib_dglsspp"
ROBUST_CKPT="robust_diagnostic/logs/med_corsupcon_21ep/supcon_vib_dglsspp_corsupcon"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

# ---- [1/3] training-side probe efficiency ----
echo ""
echo "=== [1/3] spectral-trunc (fit efficiency) ==="
CONDS="$EFF_CONDS" MAX_FRAMES="$MAX_FRAMES" \
  bash run_probe_spectral_trunc.sh "$GPU" || fail "spectral-trunc"

# ---- [2/3] inference-side probe efficiency ----
echo ""
echo "=== [2/3] decode-quant (decode efficiency) ==="
CONDS="$EFF_CONDS" MAX_FRAMES="$MAX_FRAMES" \
  bash run_probe_decode_quant.sh "$GPU" || fail "decode-quant"

# ---- [3/3] robust AL-gauge across extractors ----
echo ""
echo "=== [3/3] AL-gauge multi-extractor (label-free 'should we AL?') ==="
if [ "$DRY_RUN" = "TRUE" ]; then
  GAUGE_EXTRS="cov_ep10:${METHOD}:${CKPT}"
else
  GAUGE_EXTRS="cov_ep10:${METHOD}:${CKPT},cov_ep21:${METHOD}:${CKPT21},dglsspp:supcon_vib_dglsspp:${DGLSSPP_CKPT},robust:supcon_vib_dglsspp_corsupcon:${ROBUST_CKPT}"
fi
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/probe_al_gauge_multi_diag.py \
  --extractors "$GAUGE_EXTRS" \
  --conds "$CONDS" --max_frames "$MAX_FRAMES" \
  --label "gauge_multi${DRY_RUN:+-dry}" \
  --out "robust_diagnostic/logs/probe_al_gauge_multi${DRY_RUN:+-dry}.json" \
  2>&1 | tee "logs/probe_al_gauge_multi${DRY_RUN:+-dry}.log" || fail "al-gauge-multi"

echo ""
echo "=== OVERNIGHT PROBE-EFF COMPLETE ==="
echo "  spectral-trunc: robust_diagnostic/logs/probe_spectral_trunc_ep10.json"
echo "  decode-quant:   robust_diagnostic/logs/probe_decode_quant_ep10.json"
echo "  al-gauge-multi: robust_diagnostic/logs/probe_al_gauge_multi${DRY_RUN:+-dry}.json"
