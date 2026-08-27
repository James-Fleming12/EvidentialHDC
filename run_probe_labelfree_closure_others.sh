#!/usr/bin/env bash
# run_probe_labelfree_closure_others.sh: verify the label-free-gating CLOSURE
# holds on the OTHER feature extractors, not just cov-shift DGLSS++.
#
# The Iterations 9-12 closure (no reliable label-free gating: the probe's
# label-free ceiling IS the frozen decoder) was measured only on cov-shift
# (supcon_vib_dglsspp_inputin_in_chan). Before committing the active-learning
# framework (Pillar 3) to that closure, re-run the same four diagnostics on:
#   - HyperLiDAR default (method `baseline`, checkpoint logs/kitti_pretrain)
#   - GeoID-loss port  (method `supcon_vib_geoid`, checkpoint
#                       robust_diagnostic/logs/geoid_full/supcon_vib_geoid)
# and optionally cov-shift ep10 as an in-batch reference (INCLUDE_COV=1).
#
# The four diagnostics (one per closure iteration):
#   1. pseudo_gate        (Iter 9)  hard gates: conf/margin/norm/uncertainty
#   2. weighted_2stage    (Iter 10) soft weights + two-stage pseudo-label update
#   3. pseudolabel_struct (Iter 11) the comprehensive S/T decomposition
#   4. geometric_tta      (Iter 12) S-only Procrustes/CORAL/diffusion
#
# Usage:
#   DRY_RUN=1 bash run_probe_labelfree_closure_others.sh 2
#   SMOKE=1   bash run_probe_labelfree_closure_others.sh 2
#   bash run_probe_labelfree_closure_others.sh 2
#   CONDS="fog,crosstalk" bash run_probe_labelfree_closure_others.sh 2
#   INCLUDE_COV=1         bash run_probe_labelfree_closure_others.sh 2  # +cov reference
#
# Output (per extractor x diagnostic):
#   robust_diagnostic/logs/probe_pseudo_gate_{hyper_kitti,geoid,covshift_ep10}.json
#   robust_diagnostic/logs/probe_weighted_2stage_{...}.json
#   robust_diagnostic/logs/probe_pseudolabel_struct_{...}.json
#   robust_diagnostic/logs/probe_geometric_tta_{...}.json
#   logs/probe_{pseudo_gate,weighted_2stage,pseudolabel_struct,geometric_tta}_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
INCLUDE_COV="${INCLUDE_COV:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "Label-free closure on other extractors | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE INCLUDE_COV=$INCLUDE_COV"
echo "  conds=$CONDS"

FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

# ---- extractor specs: name | method | checkpoint dir ----
HYPER="hyper_kitti|baseline|logs/kitti_pretrain"
GEOID="geoid|supcon_vib_geoid|robust_diagnostic/logs/geoid_full/supcon_vib_geoid"
COV="covshift_ep10|supcon_vib_dglsspp_inputin_in_chan|robust_diagnostic/logs/ep10_supcon_vib_dglsspp_inputin_in_chan/supcon_vib_dglsspp_inputin_in_chan"

EXTRACTORS="$HYPER,$GEOID"
if [ "$INCLUDE_COV" = "1" ]; then
  EXTRACTORS="$EXTRACTORS,$COV"
fi

# extra args for SMOKE (tiny frames/pool so each run is seconds)
SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 2000 --val_size 4000 --max_clean 5000"
  echo "  [SMOKE] $SMOKE_ARGS"
fi

IFS=',' read -ra EXS <<< "$EXTRACTORS"
for entry in "${EXS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"
  method="${rest%%|*}"; ckpt="${rest#*|}"
  echo ""
  echo "======================================================"
  echo "=== extractor: $label (method=$method, ckpt=$ckpt) ==="
  echo "======================================================"
  for diag in pseudo_gate weighted_2stage pseudolabel_struct geometric_tta; do
    script=""
    outname=""
    case "$diag" in
      pseudo_gate)        script="probe_pseudo_gate_diag.py";        outname="probe_pseudo_gate"; ;;
      weighted_2stage)    script="probe_weighted_two_stage_diag.py"; outname="probe_weighted_2stage"; ;;
      pseudolabel_struct) script="probe_pseudolabel_structure_diag.py"; outname="probe_pseudolabel_struct"; ;;
      geometric_tta)      script="probe_geometric_tta_diag.py";      outname="probe_geometric_tta"; ;;
    esac
    logf="logs/${outname}_${label}.log"
    outjson="robust_diagnostic/logs/${outname}_${label}.json"
    echo ""
    echo "--- [$diag] $label [$CONDS] ---"
    CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/$script \
      --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
      $SMOKE_ARGS --out \"$outjson\""
    echo "  CMD: $CMD"
    if [ "$DRY_RUN" = "1" ]; then
      echo "  [DRY] not executed"
      continue
    fi
    if eval "$CMD" 2>&1 | tee "$logf"; then
      echo "  [$diag] $label OK -> $outjson"
    else
      echo "  [$diag] $label FAILED -- tail of $logf:"
      tail -25 "$logf"
      fail "$diag $label"
    fi
  done
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$FAIL" = false ]; then
  echo "=== LABEL-FREE CLOSURE (OTHER EXTRACTORS) OK ==="
  echo "For each extractor (hyper_kitti, geoid[, covshift_ep10]) compare against the"
  echo "cov-shift closure in the docs:"
  echo "  pseudo_gate:        do conf/margin/norm/uncer gates climb from no_gate"
  echo "                      toward the oracle? (cov-shift: ALL stay near no_gate)"
  echo "  weighted_2stage:    do soft weights / two-stage climb? (cov-shift: flat/worse)"
  echo "  pseudolabel_struct: AUROC, S_all,T_gated vs S_gated, coverage, precision->mIoU"
  echo "                      (cov-shift: gate AUROC high but coverage kills the update)"
  echo "  geometric_tta:      spectral_overlap ~1, procrustes/CORAL/diffusion <= frozen?"
  echo "If ANY extractor shows a gate that CLIMBS toward oracle, the AL handoff"
  echo "assumption is violated for it -- the closure does NOT transfer."
else
  echo "=== LABEL-FREE CLOSURE FAILED ==="
  exit 1
fi
