#!/usr/bin/env bash
# Gate check for the neighborhood-purity regularizer (Iteration 13.2): micro-train it
# on the full method AND on the AL-cleanest nocons base, then verify the MECHANISM
# before a medium run.
#
# Safety indicators (vs the corsupcon micro reference):
#   - nn1 (1-NN same-class purity) UP on the nnpull variants  -> the regularizer
#     actually raises the property it targets (which drives the ceiling + AL).
#   - crosstalk naive gap held (~>= 0.3) and fog gap not worse -> TTA is kept.
#
# Usage:
#   bash run_nnpull_gate.sh            # GPU 3

set -u
GPU="${1:-3}"
echo "Using GPU $GPU"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_abl() {
  local method="$1"; local logdir="$2"; local label="$3"
  echo "=== [$label] micro training (12 ep / 10% data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs 12 --cutoff 0.1 --log_dir "$logdir" \
    2>&1 | tee "logs/dglsspp_${label}_micro.log" || fail "train $label"
  echo "=== [$label] scale_gap per-class autopsy ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/scale_gap_diag.py \
    --method "$method" --path "$logdir/$method" --label "${label}_micro" \
    2>&1 | tee "logs/dglsspp_${label}_micro_diag.log" || fail "eval $label"
}

run_abl "supcon_vib_dglsspp_corsupcon_nnpull" "robust_diagnostic/logs/micro_nnpull" "corsupcon_nnpull"
run_abl "supcon_vib_dglsspp_corsupcon_nocons_nnpull" "robust_diagnostic/logs/micro_nnpull_nocons" "corsupcon_nocons_nnpull"

echo "=== AL readiness (nn1): corsupcon ref vs nnpull vs nocons_nnpull ==="
CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_readiness_diag.py \
  --checkpoints \
"corsupcon_ref:supcon_vib_dglsspp_corsupcon:robust_diagnostic/logs/micro_corsupcon/supcon_vib_dglsspp_corsupcon,\
nnpull:supcon_vib_dglsspp_corsupcon_nnpull:robust_diagnostic/logs/micro_nnpull/supcon_vib_dglsspp_corsupcon_nnpull,\
nocons_nnpull:supcon_vib_dglsspp_corsupcon_nocons_nnpull:robust_diagnostic/logs/micro_nnpull_nocons/supcon_vib_dglsspp_corsupcon_nocons_nnpull" \
  --out "robust_diagnostic/logs/al_readiness_nnpull_gate.json" \
  2>&1 | tee "logs/al_readiness_nnpull_gate.log" || fail "al readiness"

echo ""
echo "=== GATE SUMMARY (vs corsupcon micro) ==="
echo "From logs/al_readiness_nnpull_gate.log: nn1 should be HIGHER on the nnpull"
echo "  variants (the regularizer's direct target)."
echo "From logs/dglsspp_corsupcon_nnpull_micro_diag.log: crosstalk naive gap ~>= 0.3,"
echo "  fog gap not worse than the reference."
echo "If nn1 is up AND TTA holds at BOTH nnpull settings, a medium run is the next step:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods supcon_vib_dglsspp_corsupcon_nnpull --epochs 21 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_nnpull"
