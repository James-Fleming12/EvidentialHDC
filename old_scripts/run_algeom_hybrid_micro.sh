#!/usr/bin/env bash
# run_algeom_hybrid_micro.sh: the C18 hybrid micro sweep -- the ball/spec
# AL-geometry losses on the COV-SHIFT base (README 6.1 option 3).
# C18 showed the two extractors' strengths are disjoint: cov-shift has the HIGH
# fog/crosstalk ceilings but cheap AL buys ~nothing there (frozen already near
# its ceiling); corsupcon ball/spec are AL-friendly (+0.05..0.08 at 32 labels)
# but have ~0.18 lower fog/crosstalk ceilings. These variants add ONLY ball/spec
# to the cov-shift recipe (GMSIFC/LSCC kept untouched), testing whether ONE FE
# gets both: the cov-shift ceilings AND the AL-friendly geometry.
#
#   supcon_vib_dglsspp_inputin_in_chan_ball      : + intra-class ball tightening
#   supcon_vib_dglsspp_inputin_in_chan_spec      : + covariance spectrum flatten
#   supcon_vib_dglsspp_inputin_in_chan_ball_spec : both at half weights
#   (plus the cov-shift base itself as the reference arm)
#
# Gate: al_geometry_eval (intra/kappa/prank/10-COMB AL at 64-72 labels) +
# cond_structure vs plain DGLSS++ (the cov-shift ceiling/regression check).
#
# Usage:
#   bash run_algeom_hybrid_micro.sh 3                      # all 4 arms, 8 ep/10%
#   bash run_algeom_hybrid_micro.sh 3 ball,spec            # subset
#   bash run_algeom_hybrid_micro.sh 3 ball,spec gate       # gate only (trained)
#   SKIP_EVAL=1 bash run_algeom_hybrid_micro.sh 3 ball     # train only

set -u
set -o pipefail
GPU="${1:-3}"
VARIANTS="${2:-base,ball,spec,ball_spec}"
MODE="${3:-train}"
EPOCHS=8
CUTOFF=0.1
SKIP_EVAL="${SKIP_EVAL:-0}"
echo "Using GPU $GPU, $EPOCHS ep / $CUTOFF cutoff, variants=$VARIANTS, mode=$MODE, SKIP_EVAL=$SKIP_EVAL"

BASE="supcon_vib_dglsspp_inputin_in_chan"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

IFS=',' read -ra VAR_LIST <<< "$VARIANTS"
TRAIN_FLAG=""
if [ "$MODE" = "resume" ]; then
  TRAIN_FLAG="--resume"
fi

run_one() {
  local suffix="$1"; local label="$2"
  local method
  if [ "$suffix" = "base" ]; then
    method="${BASE}"
  else
    method="${BASE}_${suffix}"
  fi
  local ckpt_dir="robust_diagnostic/logs/micro_hybrid_$label/$method"
  echo ""
  if [ "$MODE" = "gate" ]; then
    echo "=== [$label] gate only (checkpoint already trained) ==="
    if [ ! -f "$ckpt_dir/SENet" ]; then
      echo "ERROR: no checkpoint at $ckpt_dir/SENet -- run mode 'train' or 'resume' first" >&2
      FAIL=true
      return 1
    fi
  else
    echo "=== [$label] micro training ($EPOCHS ep / $CUTOFF cutoff, mode=$MODE) ==="
    CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
      --methods "$method" --epochs "$EPOCHS" --cutoff "$CUTOFF" $TRAIN_FLAG \
      --log_dir "robust_diagnostic/logs/micro_hybrid_$label" \
      2>&1 | tee "logs/micro_hybrid_${label}_train.log" || fail "train $label"
  fi

  echo "=== [$label] AL-geometry gate ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b "$ckpt_dir" --method_b "$method" --label_b "hybrid_$label" \
    --conds fog,crosstalk,snow,wet_ground \
    --out "robust_diagnostic/logs/algeom_gate_hybrid_$label.json" \
    2>&1 | tee "logs/algeom_gate_hybrid_$label.log" || fail "algeom gate $label"

  echo "=== [$label] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$ckpt_dir" \
    --method_b "$method" --label_b "hybrid_$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/algeom_cond_gate_hybrid_$label.json" \
    2>&1 | tee "logs/algeom_cond_gate_hybrid_$label.log" || fail "cond gate $label"
}

for s in "${VAR_LIST[@]}"; do
  run_one "$s" "$s"
done

echo ""
if [ "$FAIL" = false ]; then
  echo "=== HYBRID MICRO OK ==="
  echo "Compare algeom_gate_hybrid_{base,ball,spec,ball_spec}.json:"
  echo "  - CEILING check (the cov-shift property): frozen/oracle/spec-ceil on"
  echo "    fog/crosstalk vs the pure cov-shift base arm -- must NOT regress"
  echo "    (that is the whole reason for the cov-shift architecture)."
  echo "  - AL check (the ball/spec property): 10-COMB delta at 64-72 labels"
  echo "    should turn POSITIVE on fog/crosstalk (vs the cov-shift base ~0)."
  echo "  - cond_structure: no regression on snow/wet_ground."
  echo "If both hold -> the hybrid is the ONE extractor with both properties;"
  echo "promote to medium: bash run_algeom_hybrid_medium.sh 3"
else
  echo "=== HYBRID MICRO FAILED ==="
  exit 1
fi
