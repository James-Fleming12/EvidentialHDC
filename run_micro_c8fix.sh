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
# Usage:
#   bash run_micro_c8fix.sh 3            # GPU 3, 8 ep / 10%
#   bash run_micro_c8fix.sh 3 8 0.1      # GPU 3, 8 epochs, 10% data
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein   # subset
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein,scalereg resume   # continue training
#   bash run_micro_c8fix.sh 3 8 0.1 scope,scalein,scalereg gate     # skip training, gate only

set -u
GPU="${1:-3}"
EPOCHS="${2:-8}"
CUTOFF="${3:-0.1}"
VARIANTS="${4:-scope,scalein,scalereg}"
MODE="${5:-train}"
echo "Using GPU $GPU, $EPOCHS ep / $CUTOFF cutoff, mode=$MODE"

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

echo ""
echo "=== C8 LEVER VERDICT ==="
echo "For each variant, compare vs the C6/C8 cov-shift baseline (from the ep10 run):"
echo "  - On snow/wet_ground: does corr_tight_B and zs_B recover toward the plain"
echo "    DGLSS++ (A) level, i.e. the C6 packing-loss signature is reduced?"
echo "  - On fog/crosstalk: does zs_B stay at/near the cov-shift gain (no regression)?"
echo "  The variant that recovers the healthy packing WITHOUT losing fog/crosstalk is"
echo "  the winner -> promote to the medium run:"
echo "    bash run_covshift_medium.sh 3 <method>"
