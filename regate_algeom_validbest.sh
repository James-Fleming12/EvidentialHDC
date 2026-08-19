#!/usr/bin/env bash
# regate_algeom_validbest.sh: re-run ONLY the valid_best gates for ball/spec after
# the medium run. The valid_best gates failed in run_algeom_medium_seq.sh because
# the symlink to SENet_valid_best was fragile; this uses a COPY instead. Training
# is already done -- this only gates the best-val checkpoints.
#
# Usage:
#   bash regate_algeom_validbest.sh 3      # GPU 3, gate valid_best for ball+spec

set -u
set -o pipefail
GPU="${1:-3}"
echo "Using GPU $GPU (valid_best gates only)"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"
FAIL=false
fail() { echo "ERROR: $1 failed" >&2; FAIL=true; }

run_validbest() {
  local method="$1"; local label="$2"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  if [ ! -f "$ckpt_dir/SENet_valid_best" ]; then
    echo "=== [$label:valid_best] SKIP: no SENet_valid_best ==="
    return 0
  fi
  local vdir="$ckpt_dir/valid_best"
  mkdir -p "$vdir"
  cp -f "$ckpt_dir/SENet_valid_best" "$vdir/SENet"
  local out_tag="${label}_valid_best"
  echo ""
  echo "=== [$label:valid_best] AL-geometry gate ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b "$vdir" --method_b "$method" --label_b "$out_tag" \
    --conds fog,crosstalk,snow,wet_ground \
    --out "robust_diagnostic/logs/algeom_gate_med_$out_tag.json" \
    2>&1 | tee "logs/algeom_gate_med_$out_tag.log" || fail "algeom gate $label:valid_best"

  echo "=== [$label:valid_best] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$vdir" \
    --method_b "$method" --label_b "$out_tag" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/algeom_cond_gate_med_$out_tag.json" \
    2>&1 | tee "logs/algeom_cond_gate_med_$out_tag.log" || fail "cond gate $label:valid_best"
}

run_validbest "$BALL" "ball"
run_validbest "$SPEC" "spec"

echo ""
echo "=== VALID_BEST GATES DONE ==="
echo "Compare algeom_gate_med_{ball,spec}_valid_best.json vs the _final gates:"
echo "  - if valid_best == final (same epoch): the model was still climbing at the"
echo "    end (best-val set at final epoch) -> resume 2-3 epochs."
echo "  - if valid_best is earlier and its gates are close to final: plateaued"
echo "    within the run (but still high-LR, see convergence note in the main script)."
if [ "$FAIL" = true ]; then
  echo "=== SOME GATES FAILED ==="
  exit 1
fi
