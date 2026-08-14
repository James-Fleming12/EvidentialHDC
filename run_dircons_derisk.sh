#!/usr/bin/env bash
# FAST derisk gate for the 10h dircons medium re-run (~30-40 min, single variant).
#
# The medium dircons (Iteration 19) raised the crosstalk ceiling (0.203 vs DGLSS++
# 0.214) but did not reach the classes that never shifted (car corr_dir ~0.92).
# The primary fix is dir_w 0.1 -> 0.2 (stronger displacement-direction pull). This
# derisk just checks the mechanism moves at micro before the overnight run:
#
#   dircons_w02 : dir_w 0.2, all classes
#
# Fast defaults (override with env):
#   DERISK_EPOCHS=8  DERISK_CUTOFF=0.1  DERISK_FRAMES=50  DERISK_POOL=50000
#   + optional second variant: DERISK_EXTRA=<method>  (e.g. ..._dircons_frag)
#
# Gated on the extractor_diff per-branch signals (--inv_ch 128, reduced frames/pool):
#   PASS (vs corsupcon micro ref):
#     - car corr_dir < 1      (the w02 pull reaches the class that never shifted)
#     - car fog oracle UP     (ceiling moves)
#     - inv_feat_cos high     (anchor intact)
#     - corr_tightness not collapsed
#
# Usage:
#   bash run_dircons_derisk.sh                     # GPU 3, fast single variant
#   DERISK_EXTRA=supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_frag \
#     bash run_dircons_derisk.sh 3                 # add the fragile-only variant (~+25 min)

set -u
GPU="${1:-3}"
EPOCHS="${DERISK_EPOCHS:-8}"
CUTOFF="${DERISK_CUTOFF:-0.1}"
FRAMES="${DERISK_FRAMES:-50}"
POOL="${DERISK_POOL:-50000}"
echo "Using GPU $GPU (epochs=$EPOCHS cutoff=$CUTOFF frames=$FRAMES pool=$POOL)"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
W02="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons_w02"
VARIANTS="$W02 ${DERISK_EXTRA:-}"

fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_abl() {
  local method="$1"; local logdir="$2"; local label="$3"
  echo "=== [$label] micro training (${EPOCHS} ep / ${CUTOFF} data) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff "$CUTOFF" --log_dir "$logdir" \
    2>&1 | tee "logs/derisk_${label}_micro.log" || fail "train $label"
}

for v in $VARIANTS; do
  run_abl "$v" "robust_diagnostic/logs/micro_$v" "$v"
done

# mechanism gate: each variant vs the reference, per-branch structure (reduced cost)
for v in $VARIANTS; do
  echo "=== [gate] extractor_diff $v vs corsupcon micro (--inv_ch 128, fast) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "robust_diagnostic/logs/micro_$v/$v" --method_b "$v" --label_b "$v" \
    --frames "$FRAMES" --pool_size "$POOL" --val_size "$POOL" \
    --inv_ch 128 --out "robust_diagnostic/logs/derisk_gate_$v.json" \
    2>&1 | tee "logs/derisk_gate_$v.log" || fail "gate $v"
done

echo ""
echo "=== DERISK SUMMARY ==="
echo "From logs/derisk_gate_*.log, check PASS:"
echo "  dircons_w02 : car corr_dir < 1 (reaches the class that never shifted at 0.1)"
echo "                and car oracle UP -- if it moves, the 10h w02 run is the bet."
echo "From logs/derisk_*_micro.log: the isotropy clean LP / HDC-zs decode health."
echo "If it holds, its 10h medium run is a safe bet:"
echo "  uv run python robust_diagnostic/isotropy_diag.py --methods $W02 --epochs 17 --cutoff 1.0 --log_dir robust_diagnostic/logs/med_dircons_w02"
