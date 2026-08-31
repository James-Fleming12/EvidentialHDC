#!/usr/bin/env bash
# run_micro_stoch.sh: micro test of CONDITIONAL input-IN training.
#
# Motivation (cov_full_scale.md): the eval-only stats gate was FLAT -- it cannot
# recover healthy capacity because the weights were trained with input-IN always
# on (train/eval mismatch). Conditional input-IN trains the network with the
# per-scan normalization applied to a RANDOM SUBSET of scans (input_in_prob), so
# the weights support BOTH normalized and raw inputs. Then at eval the gate is a
# clean on/off the network actually knows how to use.
#
#   supcon_vib_dglsspp_inputin_in_chan_stoch   : input_in_prob = 0.5
#   supcon_vib_dglsspp_inputin_in_chan_stoch7  : input_in_prob = 0.7
#   supcon_vib_dglsspp_inputin_in_chan_stoch9  : input_in_prob = 0.9
#
# For each variant: micro-train (via isotropy_diag), then evaluate frozen mIoU
# with input-IN ON vs OFF (gate) on fog/crosstalk (rescue must stay when ON) and
# snow/wet (capacity must recover when OFF).
#
# Usage:
#   bash run_micro_stoch.sh 2                 # 8 ep / 10%, all three variants
#   bash run_micro_stoch.sh 2 8 0.1 stoch7    # subset
#
# Output: robust_diagnostic/logs/micro_stoch_gate_<variant>.json

set -u
set -o pipefail
GPU="${1:-2}"
EPOCHS="${2:-8}"
CUTOFF="${3:-0.1}"
VARIANTS="${4:-stoch,stoch7,stoch9}"
echo "Using GPU $GPU, $EPOCHS ep / $CUTOFF cutoff, variants=$VARIANTS"

BASE="supcon_vib_dglsspp_inputin_in_chan"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }
IFS=',' read -ra VAR_LIST <<< "$VARIANTS"

run_one() {
  local suffix="$1"; local label="$1"
  local method="${BASE}_${suffix}"
  local ckpt_dir="robust_diagnostic/logs/micro_stoch_$label/$method"
  echo ""
  echo "=== [$label] micro conditional-input-IN training ($EPOCHS ep / $CUTOFF cutoff) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff "$CUTOFF" \
    --log_dir "robust_diagnostic/logs/micro_stoch_$label" \
    2>&1 | tee "logs/micro_stoch_${label}_train.log" || fail "train $label"

  echo "=== [$label] gate: frozen with input-IN ON vs OFF (fog/crosstalk/snow/wet) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/eval_stoch_gate_diag.py \
    --ckpt_dir "$ckpt_dir" \
    --method "$method" \
    --conds snow,wet_ground,fog,crosstalk \
    --out "robust_diagnostic/logs/micro_stoch_gate_$label.json" \
    2>&1 | tee "logs/micro_stoch_gate_$label.log" || fail "gate $label"
}

for s in "${VAR_LIST[@]}"; do
  run_one "$s"
done

echo ""
if [ "$FAIL" = false ]; then
  echo "=== MICRO STOCH OK ==="
  echo "Compare micro_stoch_gate_<v>.json:"
  echo "  - fog/crosstalk: does frozen with input-IN ON stay at the cov-shift rescue?"
  echo "  - snow/wet: does frozen with input-IN OFF recover toward plain DGLSS++?"
  echo "  If BOTH hold, the conditional-input-IN gate works where the eval-only gate failed."
else
  echo "=== MICRO STOCH FAILED ==="
  exit 1
fi
