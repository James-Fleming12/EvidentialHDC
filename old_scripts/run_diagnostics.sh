#!/usr/bin/env bash
# Sequential diagnostic run (~2h): query gates on both existing strongvib
# checkpoints, then the midvib step-budget probe (train + ladder).
#
# Usage:
#   ./run_diagnostics.sh [GPU_ID]     # default GPU 3
#   nohup ./run_diagnostics.sh 3 > logs/diagnostics_driver.log 2>&1 &
#
# Decision readouts when it finishes:
#   - logs/oracle_gating_qg_micro.log / _med.log : Query Gate block per condition
#       (does mIoU jump between tau=inf and tau=4-6 with sane retention?)
#   - logs/midvib_probe_train.log : Deep diagnostics (clean L2 must stay ~4-6,
#       NOT collapse toward 2.4)
#   - logs/oracle_gating_midvib_probe.log : clean zero-shot must stay high (not 43.7%)
set -euo pipefail

GPU="${1:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG_DIR="logs"
T0=$(date +%s)

run_step() {
    local name="$1"; shift
    local log="$LOG_DIR/$name.log"
    echo ""
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] $name -> $log"
    echo "============================================================"
    if ! uv run python "$@" 2>&1 | tee "$log"; then
        echo "FAILED: $name (see $log)" >&2
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] done: $name"
}

# --- Step 1: query gate on the best micro-30ep strongvib encoder ---
MICRO_LOAD="logs/micro_pretrain_long/supcon_vib_strongvib"
[ -d "$MICRO_LOAD" ] || { echo "Missing checkpoint: $MICRO_LOAD"; exit 1; }
run_step oracle_gating_qg_micro oracle_gating_eval.py \
    --load_path "$MICRO_LOAD" \
    --method supcon_vib_strongvib

# --- Step 2: query gate on the collapsed medium-26ep strongvib encoder ---
MED_LOAD="logs/med_pretrain_supcon_vib_strongvib"
[ -d "$MED_LOAD" ] || { echo "Missing checkpoint: $MED_LOAD"; exit 1; }
run_step oracle_gating_qg_med oracle_gating_eval.py \
    --load_path "$MED_LOAD" \
    --method supcon_vib_strongvib

# --- Step 3: midvib (KL 0.03) step-budget probe: 8 epochs x 50% data ~ 12.7k steps ---
run_step midvib_probe_train micro_pretrain_eval.py \
    --methods supcon_vib_midvib \
    --epochs 8 \
    --cutoff 0.5 \
    --out_dir logs/micro_pretrain_midvib_probe

# --- Step 4: v4 ladder on the midvib probe checkpoint ---
MIDVIB_LOAD="logs/micro_pretrain_midvib_probe/supcon_vib_midvib"
[ -d "$MIDVIB_LOAD" ] || { echo "Missing checkpoint: $MIDVIB_LOAD"; exit 1; }
run_step oracle_gating_midvib_probe oracle_gating_eval.py \
    --load_path "$MIDVIB_LOAD" \
    --method supcon_vib_midvib

echo ""
echo "All steps done in $(( ($(date +%s) - T0) / 60 )) min"
