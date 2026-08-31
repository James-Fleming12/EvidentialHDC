#!/usr/bin/env bash
# run_algeom_medium.sh: promote ball/spec to the medium run (10 ep / 100% data).
# The micro C12 sweep (run_al_geometry_train.sh) validated that ball tightens the
# class balls and spec flattens the covariance spectrum, but the micro-scale
# 10-COMB AL curve is not a trustworthy AL verdict (it goes negative even for the
# base). This is the scale check: the Iteration-10 AL win was at MEDIUM scale, so
# this measures the AL curve where it matters, plus the property/ceiling gates.
#
# The two arms train in PARALLEL on two GPUs (default 3 + 4) so both finish in
# ~5h at 100% data (each arm ~5h, mirroring run_covshift_medium.sh scale); then
# the AL-geometry + cond_structure gates run on both.
#
# Usage:
#   bash run_algeom_medium.sh 3 4                # ball->GPU3, spec->GPU4, 10 ep / 100%
#   bash run_algeom_medium.sh 3 4 12             # custom epochs
#   bash run_algeom_medium.sh 3 4 12 resume      # continue training to target epochs
#   SKIP_EVAL=1 bash run_algeom_medium.sh 3 4 10 # train only, skip the gates

set -u
set -o pipefail
GPU_A="${1:-3}"
GPU_B="${2:-4}"
EPOCHS="${3:-10}"
MODE="${4:-train}"
SKIP_EVAL="${SKIP_EVAL:-0}"
echo "Using GPU $GPU_A (ball) + GPU $GPU_B (spec), $EPOCHS ep / 100% data, mode=$MODE, SKIP_EVAL=$SKIP_EVAL"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"

TRAIN_FLAG=""
if [ "$MODE" = "resume" ]; then
  TRAIN_FLAG="--resume"
fi

run_train() {
  local method="$1"; local gpu="$2"; local label="$3"
  echo ""
  echo "=== [$label] medium training ($EPOCHS ep / 100% data) on GPU $gpu ==="
  CUDA_VISIBLE_DEVICES=$gpu uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff 1.0 $TRAIN_FLAG \
    --log_dir "robust_diagnostic/logs/med_algeom_$label" \
    2>&1 | tee "logs/med_algeom_${label}_train.log"
}

run_train "$BALL" "$GPU_A" "ball" &
PID_A=$!
run_train "$SPEC" "$GPU_B" "spec" &
PID_B=$!

echo "Training in parallel: ball (GPU $GPU_A, PID $PID_A), spec (GPU $GPU_B, PID $PID_B)"
wait $PID_A || echo "ERROR: ball training failed (exit $?)" >&2
wait $PID_B || echo "ERROR: spec training failed (exit $?)" >&2
echo "=== Both training runs finished ==="

if [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=1: skipping gates"
  exit 0
fi

FAIL=false
fail() { echo "ERROR: $1 failed" >&2; FAIL=true; }

run_gate() {
  local method="$1"; local label="$2"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  echo ""
  echo "=== [$label] AL-geometry gate ==="
  CUDA_VISIBLE_DEVICES=$GPU_A uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b "$ckpt_dir" --method_b "$method" --label_b "med_$label" \
    --conds fog,crosstalk,snow,wet_ground \
    --out "robust_diagnostic/logs/algeom_gate_med_$label.json" \
    2>&1 | tee "logs/algeom_gate_med_$label.log" || fail "algeom gate $label"

  echo "=== [$label] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU_A uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$ckpt_dir" \
    --method_b "$method" --label_b "med_$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/algeom_cond_gate_med_$label.json" \
    2>&1 | tee "logs/algeom_cond_gate_med_$label.log" || fail "cond gate $label"
}

run_gate "$BALL" "ball"
run_gate "$SPEC" "spec"

echo ""
echo "=== MEDIUM AL-GEOMETRY CHECK ==="
echo "Compare algeom_gate_med_{ball,spec}.json vs the micro C12 numbers AND vs the"
echo "Iteration-10 medium-scale baseline (README 4.5 / active_iterations.md):"
echo "  - property gains (intra-cos, kappa, prank) should survive the scale-up;"
echo "  - the 10-COMB AL curve should now be POSITIVE where micro was negative"
echo "    (the Iteration-10 win was wet_ground/fog at medium scale);"
echo "  - frozen / oracle / spec-ceil should not regress vs the base extractor."
echo "If the AL curve still needs convergence:"
echo "  bash run_algeom_medium.sh $GPU_A $GPU_B $((EPOCHS+5)) resume"
if [ "$FAIL" = true ]; then
  echo "=== SOME GATES FAILED ==="
  exit 1
fi
