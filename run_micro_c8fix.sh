#!/usr/bin/env bash
# Iteration-C8 micro sweep: the three training-side levers for the cov-shift
# healthy-condition ceiling loss. C8 proved the loss is CONTINUOUS (survives every
# decoding: sign/bias/zscore/fourier all lose it equally), so the fix must be
# training-side, not a decoder projection/binarization change.
#
#   supcon_vib_dglsspp_inputin_in_chan_scope    : InstanceNorm only in the late stages
#                                                 (layer3/4 + bottleneck conv_1/2); the
#                                                 early geometry blocks keep BatchNorm so
#                                                 the healthy conditions' early-stage
#                                                 per-dimension anisotropy survives.
#   supcon_vib_dglsspp_inputin_in_chan_scalein  : scale-only internal InstanceNorm
#                                                 (divide by per-scan per-channel std,
#                                                 no centering) preserving the
#                                                 per-dimension offset structure.
#   supcon_vib_dglsspp_inputin_in_chan_scalereg : feature-scale regularizer in the
#                                                 trainer (clean-view z8 per-dim std
#                                                 pulled toward its EMA) so InstanceNorm
#                                                 cannot drift the healthy feature scale.
#
# Gate: cond_structure_diag per variant vs the plain-DGLSS++ baseline, measuring
# corr_tight (the C6 packing-loss metric) + zs on the healthy conditions (snow,
# wet_ground) -- the packing-recovery check -- AND on fog/crosstalk -- the
# no-regression check. The winner gets promoted to the medium run.
#
# Overnight stages (run after the variant loop, eval-only on the ep10/ep21 cov-shift
# weights):
#   stage 2 (hdc_rule): the C8 decision-rule diagnostic -- per-class scaled cosine vs
#     learned 128-d probe on frozen features (snow/wet_ground/fog/crosstalk).
#   stage 3 (nusc): KITTI weights -> NuScenes zero-shot + oracle, checking for a
#     cross-domain TTA gap (oracle - zs) that did not exist on KITTI.
#
# Usage:
#   bash run_micro_c8fix.sh 3            # GPU 3, 8 ep / 10% + overnight stages
#   bash run_micro_c8fix.sh 3 8 0.1      # GPU 3, 8 epochs, 10% data + overnight stages
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein   # subset
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein,scalereg resume   # continue training
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein,scalereg gate     # skip training, gate only

set -u
set -o pipefail
GPU="${1:-3}"
EPOCHS="${2:-8}"
CUTOFF="${3:-0.1}"
VARIANTS="${4:-scope,scalein,scalereg}"
MODE="${5:-train}"
NUSC_DIR="${NUSC_DIR:-/mnt/alpha/jmfleming/nuscenes_kitti}"
echo "Using GPU $GPU, $EPOCHS ep / $CUTOFF cutoff, mode=$MODE, NuScenes=$NUSC_DIR"

BASE="supcon_vib_dglsspp_inputin_in_chan"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"     # plain DGLSS++ medium
DGLSSPP_METHOD="supcon_vib_dglsspp"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

# split the comma-separated variant list properly (whitespace splitting is the bug
# that turned 'scope,scalein,scalereg' into one garbage method name)
IFS=',' read -ra VAR_LIST <<< "$VARIANTS"

TRAIN_FLAG=""
if [ "$MODE" = "resume" ]; then
  TRAIN_FLAG="--resume"
fi

run_one() {
  local suffix="$1"; local label="$2"
  local method="${BASE}_${suffix}"
  local ckpt_dir="robust_diagnostic/logs/micro_c8_$label/$method"
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
      --log_dir "robust_diagnostic/logs/micro_c8_$label" \
      2>&1 | tee "logs/micro_c8_${label}_train.log" || fail "train $label"
  fi

  echo "=== [$label] cond_structure gate vs plain DGLSS++ ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$ckpt_dir" \
    --method_b "$method" --label_b "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/micro_c8_gate_$label.json" \
    2>&1 | tee "logs/micro_c8_gate_$label.log" || fail "gate $label"
}

for s in "${VAR_LIST[@]}"; do
  run_one "$s" "$s"
done

# ============================================================================
# Overnight stage 2: HDC decision-rule diagnostic on the ep10/ep21 cov-shift weights.
# C8 ruled out encoding changes; this tests the CLASS-CONDITIONAL decision rule
# (per-class scaled cosine + learned 128-d probe) on the SAME frozen features.
# ============================================================================
METHOD="supcon_vib_dglsspp_inputin_in_chan"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
EP21_CKPT="robust_diagnostic/logs/med_$METHOD/$METHOD"

run_hdc_rule() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [hdc_rule] $label: per-class scaled distance vs learned probe ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/hdc_rule_diag.py \
    --path_b "$ckpt" --method_b "$METHOD" --label_b "$label" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/hdc_rule_$label.json" \
    2>&1 | tee "logs/hdc_rule_$label.log" || fail "hdc_rule $label"
}

# ============================================================================
# Overnight stage 3: NuScenes cross-domain zero-shot + oracle of the KITTI weights.
# ============================================================================
run_nusc() {
  local ckpt="$1"; local label="$2"
  echo ""
  echo "=== [nusc] $label: KITTI weights -> NuScenes zero-shot + oracle ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/nusc_cross_domain_diag.py \
    --path "$ckpt" --method "$METHOD" --label "$label" \
    --nusc_dir "$NUSC_DIR" \
    --out "robust_diagnostic/logs/nusc_$label.json" \
    2>&1 | tee "logs/nusc_$label.log" || fail "nusc $label"
}

echo ""
echo "=== Overnight eval stages (ep10 + ep21 cov-shift weights) ==="
run_hdc_rule "$EP10_CKPT" "covshift_ep10"
run_hdc_rule "$EP21_CKPT" "covshift_ep21"
run_nusc "$EP10_CKPT" "covshift_ep10"
run_nusc "$EP21_CKPT" "covshift_ep21"

echo ""
echo "=== C8 LEVER VERDICT ==="
echo "For each variant, compare vs the C6/C8 cov-shift baseline (from the ep10 run):"
echo "  - On snow/wet_ground: does corr_tight_B and zs_B recover toward the plain"
echo "    DGLSS++ (A) level, i.e. the C6 packing-loss signature is reduced?"
echo "  - On fog/crosstalk: does zs_B stay at/near the cov-shift gain (no regression)?"
echo "  The variant that recovers the healthy packing WITHOUT losing fog/crosstalk is"
echo "  the winner -> promote to the medium run:"
echo "    bash run_covshift_medium.sh 3 <method>"
echo ""
echo "=== HDC-RULE VERDICT (logs/hdc_rule_*.log) ==="
echo "  On snow/wet_ground: does R2 (per-class scaled cosine) recover the oracle toward"
echo "  plain DGLSS++ (~0.27) WITHOUT dropping the fog/crosstalk R1 oracle? R3 (learned"
echo "  128-d probe) is the strong continuous reference."
echo ""
echo "=== NUSCENES VERDICT (logs/nusc_*.log) ==="
echo "  gap = oracle - zs on NuScenes vs on KITTI clean. A large NuScenes gap with a"
echo "  small KITTI gap = TTA headroom that did not exist before; high NuScenes lp_miou"
echo "  = the continuous features transferred and the gap is in HDC binarization."
