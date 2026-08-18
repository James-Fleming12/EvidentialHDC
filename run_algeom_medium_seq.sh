#!/usr/bin/env bash
# run_algeom_medium_seq.sh: promote ball/spec to the medium run (100% data) on ONE
# GPU, run in SEQUENCE. The micro C12 sweep validated that ball tightens the class
# balls and spec flattens the covariance spectrum, but the micro-scale 10-COMB AL
# curve is not a trustworthy AL verdict (it goes negative even for the base). This
# is the scale check: the Iteration-10 AL win was at MEDIUM scale.
#
# Timing (calibrated from the micro run: ~4.2 min/epoch at 10% -> ~31 min/epoch at
# 100%): 8 epochs/arm = ~4.2h each -> ~8.3h total. 10 epochs/arm = ~10.4h total.
#
# Usage:
#   bash run_algeom_medium_seq.sh 3                 # GPU 3, 8 ep / 100%, sequential
#   bash run_algeom_medium_seq.sh 3 10              # 10 epochs per arm (~10.4h)
#   bash run_algeom_medium_seq.sh 3 10 resume       # continue training to target epochs
#   SKIP_EVAL=1 bash run_algeom_medium_seq.sh 3 8   # train only, skip the gates

set -u
set -o pipefail
GPU="${1:-3}"
EPOCHS="${2:-8}"
MODE="${3:-train}"
SKIP_EVAL="${SKIP_EVAL:-0}"
echo "Using GPU $GPU, $EPOCHS ep / 100% data, mode=$MODE, SKIP_EVAL=$SKIP_EVAL"

BALL="supcon_vib_dglsspp_corsupcon_ball"
SPEC="supcon_vib_dglsspp_corsupcon_spec"

TRAIN_FLAG=""
if [ "$MODE" = "resume" ]; then
  TRAIN_FLAG="--resume"
fi

report_ckpt() {
  local label="$1"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/supcon_vib_dglsspp_corsupcon_$label"
  echo ""
  echo "=== [$label] checkpoint epochs ==="
  uv run python - "$ckpt_dir" <<'PY'
import sys, torch, os
d = sys.argv[1]
for name, f in [("final", "SENet"), ("best_val", "SENet_valid_best")]:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        print(f"  {name}: MISSING ({p})")
        continue
    try:
        ck = torch.load(p, map_location="cpu")
        info = ck.get("info", {})
        print(f"  {name}: epoch {ck.get('epoch')} | train_iou {info.get('train_iou')} "
              f"val_iou {info.get('valid_iou')} | best_train_iou {info.get('best_train_iou')} "
              f"best_val_iou {info.get('best_val_iou')}")
    except Exception as e:
        print(f"  {name}: load error {e}")
PY
}

run_train() {
  local method="$1"; local label="$2"
  echo ""
  echo "=== [$label] medium training ($EPOCHS ep / 100% data) on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff 1.0 $TRAIN_FLAG \
    --log_dir "robust_diagnostic/logs/med_algeom_$label" \
    2>&1 | tee "logs/med_algeom_${label}_train.log"
}

FAIL=false
fail() { echo "ERROR: $1 failed" >&2; FAIL=true; }

run_train "$BALL" "ball" || fail "train ball"
report_ckpt "ball"
run_train "$SPEC" "spec" || fail "train spec"
report_ckpt "spec"
echo "=== Both training runs finished ==="

if [ "$SKIP_EVAL" = "1" ]; then
  echo "SKIP_EVAL=1: skipping gates"
  exit 0
fi

run_gate() {
  local method="$1"; local label="$2"
  local ckpt_dir="robust_diagnostic/logs/med_algeom_$label/$method"
  local tag="${3:-final}"
  local gate_path="$ckpt_dir"
  if [ "$tag" = "valid_best" ]; then
    # the eval scripts hardcode <path>/SENet, so expose the best-val checkpoint
    # as a symlinked SENet in a sibling dir (full state dict incl. epoch/info).
    local vdir="$ckpt_dir/valid_best"
    mkdir -p "$vdir"
    ln -sf "$ckpt_dir/SENet_valid_best" "$vdir/SENet"
    gate_path="$vdir"
  fi
  local out_tag="${label}_${tag}"
  echo ""
  echo "=== [$label:$tag] AL-geometry gate ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_geometry_eval.py \
    --path_b "$gate_path" --method_b "$method" --label_b "$out_tag" \
    --conds fog,crosstalk,snow,wet_ground \
    --out "robust_diagnostic/logs/algeom_gate_med_$out_tag.json" \
    2>&1 | tee "logs/algeom_gate_med_$out_tag.log" || fail "algeom gate $label:$tag"

  echo "=== [$label:$tag] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$gate_path" \
    --method_b "$method" --label_b "$out_tag" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/algeom_cond_gate_med_$out_tag.json" \
    2>&1 | tee "logs/algeom_cond_gate_med_$out_tag.log" || fail "cond gate $label:$tag"
}

# Gate BOTH the final checkpoint (what the log reports) and the best-val
# checkpoint (the one that would be selected by early stopping). If the best-val
# epoch is late (== final-1) the model is still climbing and 1-2 more epochs are
# warranted before any "it doesn't scale" claim.
run_gate "$BALL" "ball" "final"
run_gate "$BALL" "ball" "valid_best"
run_gate "$SPEC" "spec" "final"
run_gate "$SPEC" "spec" "valid_best"

echo ""
echo "=== CONVERGENCE CHECK (read the training logs) ==="
echo "Per-epoch 'Epoch: [k]' lines are in logs/med_algeom_{ball,spec}_train.log."
echo "The correct signal is WHICH epoch set best_val and the SLOPE of the last"
echo "epochs' val IoU -- NOT whether the final and best-val gates agree."
echo "  - valid_best epoch == final epoch, or the last 2-3 val IoUs are still"
echo "    rising: model is still climbing -> continue (bash run_algeom_medium_seq.sh"
echo "    $GPU $((EPOCHS+5)) resume)."
echo "  - valid_best epoch well before the end AND the last 2-3 val IoUs flat:"
echo "    plateaued WITHIN this run."
echo ""
echo "CRITICAL: the cosine scheduler runs on first_cycle=80 epochs (senet-2048p.yml),"
echo "so at $EPOCHS epochs the LR is still ~max (barely past the 1-ep warmup). A"
echo "plateau here is a HIGH-LR plateau, not an optimum. The AL-geometry numbers from"
echo "this run are a SCALING check (do the ball/spec property gains survive 100%"
echo "data?), NOT a final verdict. A negative 10-COMB at $EPOCHS epochs cannot"
echo "support a 'ball/spec does not work' claim -- that requires the LR-annealed"
echo "medium run (21 ep, run_covshift_medium.sh scale)."
echo ""
echo "=== MEDIUM AL-GEOMETRY CHECK ==="
echo "Compare algeom_gate_med_{ball,spec}.json vs the micro C12 numbers AND vs the"
echo "Iteration-10 medium-scale baseline (README 4.5 / active_iterations.md):"
echo "  - property gains (intra-cos, kappa, prank) should survive the scale-up;"
echo "  - the 10-COMB AL curve should now be POSITIVE where micro was negative"
echo "    (the Iteration-10 win was wet_ground/fog at medium scale);"
echo "  - frozen / oracle / spec-ceil should not regress vs the base extractor."
echo "If the AL curve still needs convergence:"
echo "  bash run_algeom_medium_seq.sh $GPU $((EPOCHS+5)) resume"
if [ "$FAIL" = true ]; then
  echo "=== SOME GATES FAILED ==="
  exit 1
fi
