#!/usr/bin/env bash
# Iteration-19.10 micro sweep: input-level covariate-shift normalization.
# Tests level-1 (input-IN) alone vs the level-1 + level-2 stack (input-IN + internal
# InstanceNorm), the training-side mirror of the BN-alignment TTA lever that was our
# best TTA method. Iteration-19.11 added the channel-restricted variant (the fog fix).
#
#   supcon_vib_dglsspp_inputin           : per-scan input normalization only (internal BN)
#   supcon_vib_dglsspp_inputin_in        : input-IN + internal InstanceNorm (both levels)
#   supcon_vib_dglsspp_inputin_in_chan   : the 19.11.2 fix: input-IN restricted to
#                                          range+remission channels, xyz geometry left
#                                          untouched (fog's shift survives)
#   supcon_vib_dglsspp_inputin_in_scale  : the 19.12.1 knob: scale-only (divide by
#                                          per-scan std, no mean subtraction) on all
#                                          channels -- absorbs magnitude shift, keeps
#                                          direction. The general version of _chan.
#
# Gate: extractor_diff vs the plain-DGLSS++ baseline, checking oracle (ceiling) +
# naive (TTA) on fog/crosstalk.
#
# Usage:
#   bash run_micro_inputin.sh 3            # GPU 3, 8 ep / 10%

set -u
GPU="${1:-3}"
EPOCHS="${2:-8}"
# Only the NEW variant needs running -- inputin and inputin_in were measured in the
# Iteration-19.11/19.12 sweeps (their gates are already in Logs). The scale-only knob
# is the only unmeasured one. Override with an explicit list if needed.
VARIANTS="${3:-supcon_vib_dglsspp_inputin_in_scale}"
echo "Using GPU $GPU, $EPOCHS ep / 10%"

REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"     # plain DGLSS++ medium
DGLSSPP_METHOD="supcon_vib_dglsspp"
fail() { echo "ERROR: $1 failed (exit $?)" >&2; }

run_one() {
  local method="$1"; local label="$2"
  echo "=== [$label] micro training ($EPOCHS ep / 10%) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "$method" --epochs "$EPOCHS" --cutoff 0.1 \
    --log_dir "robust_diagnostic/logs/micro_$label" \
    2>&1 | tee "logs/micro_${label}_train.log" || fail "train $label"
  echo "=== [$label] extractor_diff vs plain DGLSS++ (ceiling comparison) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
    --path_b "robust_diagnostic/logs/micro_$label/$method" --method_b "$method" --label_b "$label" \
    --frames 50 --pool_size 50000 --val_size 50000 \
    --out "robust_diagnostic/logs/micro_gate_$label.json" \
    2>&1 | tee "logs/micro_gate_$label.log" || fail "gate $label"
}

for m in $VARIANTS; do
  run_one "$m" "${m##*_}"
done

echo ""
echo "=== INPUT-IN VERDICT ==="
echo "Compare inputin_in_scale vs plain DGLSS++ (A) and vs the 19.12 winner"
echo "(inputin_in_chan, already in Logs as micro_gate_inputin_in_chan.json):"
echo "  - scale-only (no mean-shift): does it match or beat _chan on BOTH conditions?"
echo "    If so it is the simpler, more general paper story (no channel selection)."
echo "Promote the winner to the full medium run:"
echo "  bash run_covshift_medium.sh 3 <method>"
