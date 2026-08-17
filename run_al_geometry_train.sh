#!/usr/bin/env bash
# run_al_geometry_train.sh: train the feature geometry that AL/TTA needs, and
# measure it. The AL thread measured the bottlenecks as TRAINABLE properties:
#   - the fat-blob geometry (intra-cos 0.62-0.70) drives the mean-estimation
#     sample complexity, the R1-prototype viability, and T-error amplification;
#   - the ill-conditioned covariance (4-6x ridge-relevant error, fractional
#     update needing beta<1) is the inverse-covariance amplification.
# These objectives train against those properties directly:
#   base                    : the robust corsupcon base (control, already have)
#   supcon_vib_dglsspp_corsupcon_ball     : intra-class ball tightening
#   supcon_vib_dglsspp_corsupcon_spec     : covariance condition-number penalty
#   supcon_vib_dglsspp_corsupcon_ball_spec: both at half weights
#   supcon_vib_dglsspp_corsupcon_nnpull   : 1-NN purity (existing AL lever)
# Each micro-run (8 ep / 10%) is gated on the feature-space properties:
#   - AL geometry gate (al_geometry_eval): intra/inter cos, gain quantiles,
#     participation rank, mean-k curve, 1-NN purity, and the 10-COMB
#     fractional-residual AL update at 64-72 labels vs frozen/ceiling.
#   - cond_structure gate vs plain DGLSS++ (no regression on the healthy
#     conditions).
#
# Usage:
#   bash run_al_geometry_train.sh 3                  # all variants, 8ep/10%
#   bash run_al_geometry_train.sh 3 ball,spec        # subset
#   SKIP_EVAL=1 bash run_al_geometry_train.sh 3 ball gate   # gate only

set -u
set -o pipefail
GPU="${1:-3}"
VARIANTS="${2:-base,ball,spec,ball_spec,nnpull}"
MODE="${3:-train}"
EPOCHS=8
CUTOFF=0.1
SKIP_EVAL="${SKIP_EVAL:-0}"
echo "Using GPU $GPU, $EPOCHS ep / $CUTOFF cutoff, variants=$VARIANTS, mode=$MODE, SKIP_EVAL=$SKIP_EVAL"

BASE="supcon_vib_dglsspp_corsupcon"
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
  local ckpt_dir="robust_diagnostic/logs/micro_algeom_$label/$method"
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
      --log_dir "robust_diagnostic/logs/micro_algeom_$label" \
      2>&1 | tee "logs/micro_algeom_${label}_train.log" || fail "train $label"
  fi

  echo "=== [$label] AL-geometry gate ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b "$ckpt_dir" --method_b "$method" --label_b "$label" \
    --conds fog,crosstalk,snow,wet_ground \
    --out "robust_diagnostic/logs/algeom_gate_$label.json" \
    2>&1 | tee "logs/algeom_gate_$label.log" || fail "algeom gate $label"

  echo "=== [$label] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$ckpt_dir" \
    --method_b "$method" --label_b "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/algeom_cond_gate_$label.json" \
    2>&1 | tee "logs/algeom_cond_gate_$label.log" || fail "cond gate $label"
}

for s in "${VAR_LIST[@]}"; do
  run_one "$s" "$s"
done

echo ""
if [ "$FAIL" = false ]; then
  echo "=== AL-GEOMETRY-TRAIN OK ==="
  echo "Check logs/algeom_gate_*.log: intra/inter cos, gain quantiles,"
  echo "participation rank, mean-k curve, 1-NN purity, 10-COMB at 64-72 labels"
  echo "vs frozen/ceiling. The winner gets promoted to a medium run and the"
  echo "full AL budget curve."
else
  echo "=== AL-GEOMETRY-TRAIN FAILED ==="
  exit 1
fi
