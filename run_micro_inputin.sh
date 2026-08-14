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
#
# Gate: extractor_diff vs the plain-DGLSS++ baseline, checking oracle (ceiling) +
# naive (TTA) on fog/crosstalk.
#
# Usage:
#   bash run_micro_inputin.sh 3            # GPU 3, 8 ep / 10%

set -u
GPU="${1:-3}"
EPOCHS="${2:-8}"
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

run_one "supcon_vib_dglsspp_inputin" "inputin"
run_one "supcon_vib_dglsspp_inputin_in" "inputin_in"
run_one "supcon_vib_dglsspp_inputin_in_chan" "inputin_in_chan"

echo ""
echo "=== INPUT-IN VERDICT ==="
echo "From logs/micro_gate_*.log, compare each variant vs plain DGLSS++ (A):"
echo "  - oracle (ceiling) on fog/crosstalk: does per-scan input norm raise it?"
echo "  - naive (TTA): does it hold or improve?"
echo "  - inputin_in vs inputin_in_chan: does the channel-restricted version keep"
echo "    the crosstalk gain while recovering the fog oracle (the 19.11.2 fix)?"
echo "Any variant that raises the oracle without hurting naive -> promote to medium-lite."
