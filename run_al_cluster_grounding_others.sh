#!/usr/bin/env bash
# run_al_cluster_grounding_others.sh: which feature space is cheapest for AL
# oracle label reuse? Run the cluster-grounding / label-budget diagnostic
# (al_cluster_grounding_diag.py) on the OTHER feature extractors, not just
# cov-shift:
#   - HyperLiDAR default (method `baseline`, checkpoint logs/kitti_pretrain)
#   - GeoID-loss port  (method `supcon_vib_geoid`, checkpoint
#                       robust_diagnostic/logs/geoid_full/supcon_vib_geoid)
# and optionally cov-shift ep10 as an in-batch reference (INCLUDE_COV=1).
#
# This answers: do hyper/geoid have a BETTER (or worse) feature space for cheap
# AL label reuse -- tighter same-class packing (higher NN purity), cleaner
# cluster boundaries (higher distance-gated coverage), a smaller label budget
# (K -> coverage curve), and DIFFERENT weak points (which per-class clusters
# are loose on each extractor)?
#
#   A. packing:     pool 1-NN/k-NN purity, intra/inter cosine separation,
#                   k-means cluster purity at K=#classes (per class)
#   B. grounding:   budget(K in {17,34,68,136,272}) -> coverage; distance-gated
#                   coverage (q 0.5/0.75/0.9); radius for 90% coverage
#   C. reduction:   per-class shift alignment, confidence-representativeness,
#                   within-class multi-modality, pseudo-acc vs dist-to-centroid
#
# Usage:
#   DRY_RUN=1 bash run_al_cluster_grounding_others.sh 2
#   SMOKE=1   bash run_al_cluster_grounding_others.sh 2
#   bash run_al_cluster_grounding_others.sh 2
#   CONDS="fog,crosstalk"   bash run_al_cluster_grounding_others.sh 2
#   INCLUDE_COV=1           bash run_al_cluster_grounding_others.sh 2  # +cov reference
#   EXTRACTORS_OVERRIDE="geoid|supcon_vib_geoid|robust_diagnostic/logs/geoid_full/supcon_vib_geoid" \
#                         bash run_al_cluster_grounding_others.sh 2     # geoid only
#
# Output (per extractor):
#   robust_diagnostic/logs/al_cluster_grounding_{hyper_kitti,geoid,covshift_ep10}.json
#   logs/al_cluster_grounding_{label}.log

set -u
set -o pipefail
GPU="${1:-2}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
INCLUDE_COV="${INCLUDE_COV:-0}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground}"
EXTRACTORS_OVERRIDE="${EXTRACTORS_OVERRIDE:-}"
SM_FRAMES="${SM_FRAMES:-10}"
echo "AL cluster grounding on other extractors | GPU $GPU | DRY_RUN=$DRY_RUN SMOKE=$SMOKE INCLUDE_COV=$INCLUDE_COV"
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
if [ -n "$EXTRACTORS_OVERRIDE" ]; then
  EXTRACTORS="$EXTRACTORS_OVERRIDE"
fi

SMOKE_ARGS=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS="--frames $SM_FRAMES --pool_size 2000 --max_clean 5000 --cluster_ks 17,34 --within_ks 2,4"
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
  logf="logs/al_cluster_grounding_${label}.log"
  outjson="robust_diagnostic/logs/al_cluster_grounding_${label}.json"
  CMD="CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_cluster_grounding_diag.py \
    --path_b \"$ckpt\" --method_b \"$method\" --label \"$label\" --conds \"$CONDS\" \
    $SMOKE_ARGS --out \"$outjson\""
  echo "  CMD: $CMD"
  if [ "$DRY_RUN" = "1" ]; then
    echo "  [DRY] not executed"
    continue
  fi
  if eval "$CMD" 2>&1 | tee "$logf"; then
    echo "  [$label] OK -> $outjson"
  else
    echo "  [$label] FAILED -- tail of $logf:"
    tail -25 "$logf"
    fail "$label"
  fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN: commands printed, nothing executed ==="
  exit 0
fi
if [ "$FAIL" = false ]; then
  echo "=== AL CLUSTER GROUNDING (OTHER EXTRACTORS) OK ==="
  echo "Compare against the cov-shift baseline (al_cluster_grounding_covshift_ep10.json):"
  echo "  packing:   pool_nn / clean_nn / separation / k-means purity per class"
  echo "  grounding: budget(K)->coverage, distance-gated coverage, radius90"
  echo "  reduction: shift alignment, confidence-representativeness, multi-modality"
  echo "Which extractor has the CHEAPEST label reuse (highest coverage at smallest K) and"
  echo "where do their weak points DIFFER (per-class loose clusters)?"
else
  echo "=== AL CLUSTER GROUNDING FAILED ==="
  exit 1
fi
