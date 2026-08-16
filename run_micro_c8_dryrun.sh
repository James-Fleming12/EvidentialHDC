#!/usr/bin/env bash
# Iteration-C8 dry run: fast smoke test of the overnight pipeline before the real
# run. Catches import / CLI / data-path / checkpoint-path errors in minutes instead
# of hours. Uses tiny frames / pool / val and a 1-epoch micro-train.
#
# Checks (mirrors the overnight run):
#   scope   : 1-epoch micro-train + isotropy eval + cond_structure gate (tiny frames)
#   hdc_rule: the HDC decision-rule diagnostic (R1/R2/R3/R4) on the ep10 weights
#   nusc    : the NuScenes cross-domain diagnostic on the ep10 weights
#
# Usage:
#   bash run_micro_c8_dryrun.sh 3            # GPU 3
#   bash run_micro_c8_dryrun.sh 3 scope,hdc_rule,nusc   # subset

set -u
GPU="${1:-3}"
STAGES="${2:-scope,hdc_rule,nusc}"
echo "Using GPU $GPU, stages=$STAGES"

BASE="supcon_vib_dglsspp_inputin_in_chan"
METHOD="$BASE"
EP10_CKPT="robust_diagnostic/logs/ep10_$METHOD/$METHOD"
NUSC_DIR="${NUSC_DIR:-/mnt/alpha/jmfleming/nuscenes_kitti}"
FAIL=false
fail() { echo "ERROR: $1 failed (exit $?)" >&2; FAIL=true; }

DRY_DIR="robust_diagnostic/logs/micro_c8_dryrun"

if [[ ",$STAGES," == *",scope,"* ]]; then
  echo ""
  echo "=== [scope] dry micro-train (1 ep / 5%) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/isotropy_diag.py \
    --methods "${BASE}_scope" --epochs 1 --cutoff 0.05 \
    --log_dir "$DRY_DIR/scope" \
    > "logs/micro_c8_dry_scope_train.log" 2>&1 || fail "scope train"

  echo "=== [scope] dry cond_structure gate (2 frames) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/cond_structure_diag.py \
    --path_b "$DRY_DIR/scope/${BASE}_scope" \
    --method_b "${BASE}_scope" --label_b "scope_dry" \
    --frames 2 --pool_size 2000 --val_size 2000 \
    --conds snow,wet_ground,fog,crosstalk \
    --out "$DRY_DIR/gate_scope.json" \
    > "logs/micro_c8_dry_scope_gate.log" 2>&1 || fail "scope gate"
fi

if [[ ",$STAGES," == *",hdc_rule,"* ]]; then
  echo ""
  echo "=== [hdc_rule] dry decision-rule diag (2 frames, tiny pool) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/hdc_rule_diag.py \
    --path_b "$EP10_CKPT" --method_b "$METHOD" --label_b "covshift_ep10_dry" \
    --frames 2 --pool_size 2000 --val_size 2000 \
    --conds snow,wet_ground,fog,crosstalk \
    --out "$DRY_DIR/hdc_rule.json" \
    > "logs/micro_c8_dry_hdc_rule.log" 2>&1 || fail "hdc_rule"
fi

if [[ ",$STAGES," == *",nusc,"* ]]; then
  echo ""
  echo "=== [nusc] dry cross-domain diag (2 frames) ==="
  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/nusc_cross_domain_diag.py \
    --path "$EP10_CKPT" --method "$METHOD" --label "covshift_ep10_dry" \
    --frames 2 --pool_size 2000 --val_size 2000 \
    --nusc_dir "$NUSC_DIR" \
    --out "$DRY_DIR/nusc.json" \
    > "logs/micro_c8_dry_nusc.log" 2>&1 || fail "nusc"
fi

echo ""
if [ "$FAIL" = false ]; then
  echo "=== DRY RUN OK: all stages completed without error ==="
  echo "Check logs: logs/micro_c8_dry_*.log"
else
  echo "=== DRY RUN FAILED: see the errors above ==="
  exit 1
fi
