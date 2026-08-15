#!/usr/bin/env bash
# Mid-training monitor for the covariate-shift medium run (inputin_in_chan).
# While run_covshift_medium.sh trains, this snapshots the rolling checkpoint
# (SENet is overwritten every epoch) and runs the extractor_diff gate on each
# snapshot vs plain DGLSS++, logging the oracle + naive per epoch so we can find
# the OPTIMAL epoch window -- critical because the family degrades past 21 ep
# (Iteration 8.1) and this extractor may peak earlier or later.
#
# Usage (in a SECOND terminal, while the medium run trains):
#   bash monitor_covshift.sh            # GPU 3, every 30 min
#   bash monitor_covshift.sh 3 1800     # custom interval (seconds)
#
# Decision rules (from logs/covshift_med_monitor.log):
#   - oracle trending UP on fog AND crosstalk through mid-training -> on track
#   - oracle PEAKS then FALLS (watch for the knee) -> note the best epoch; after
#     training, the best checkpoint can be resumed-to / used for the battery
#   - naive TTA also UP (it should, from the micro: crosstalk 0.29 vs robust 0.13)
#   - if oracle flat or below plain DGLSS++ at epoch ~15+, the cov-shift win does
#     not transfer to medium -> stop, do not commit further
#
# The rolling SENet is the epoch-N state; _train_best / _valid_best also exist.

set -u
GPU="${1:-3}"
SLEEP="${2:-1800}"
METHOD="supcon_vib_dglsspp_inputin_in_chan"
MED_DIR="robust_diagnostic/logs/med_$METHOD"
CKPT="$MED_DIR/$METHOD/SENet"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"
DGLSSPP_METHOD="supcon_vib_dglsspp"
SNAP_DIR="robust_diagnostic/logs/monitor_covshift"
mkdir -p "$SNAP_DIR"
LOG="logs/covshift_med_monitor.log"
: > "$LOG"
echo "Monitoring $CKPT every ${SLEEP}s; results -> $LOG" | tee -a "$LOG"

while [ ! -f "$CKPT" ]; do
  echo "[$(date +%H:%M)] checkpoint not present yet, waiting..." | tee -a "$LOG"
  sleep 600
done

last_ep=-1
while true; do
  ep=$(python3 -c "
import torch
try:
    print(torch.load('$CKPT', map_location='cpu')['epoch'])
except Exception:
    print('?')" 2>/dev/null)
  if [ "$ep" = "$last_ep" ]; then
    echo "[$(date +%H:%M)] epoch $ep unchanged (training done or stalled); exiting" | tee -a "$LOG"
    exit 0
  fi
  last_ep="$ep"

  snap="$SNAP_DIR/snap_ep${ep}"
  mkdir -p "$snap"
  cp "$CKPT" "$snap/SENet"
  echo "=== [$(date +%H:%M)] snapshot epoch $ep ===" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
    --path_b "$snap" --method_b "$METHOD" --label_b "covshift_ep${ep}" \
    --frames 50 --pool_size 50000 --val_size 50000 \
    --out "$SNAP_DIR/gate_ep${ep}.json" \
    2>&1 | tee -a "$LOG" || echo "  (extractor_diff on ep $ep failed - checkpoint mid-write?)" | tee -a "$LOG"

  python3 - "$SNAP_DIR/gate_ep${ep}.json" "covshift_ep${ep}" <<'EOF' 2>/dev/null | tee -a "$LOG"
import json, sys
fn, label = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(fn))
    v = d[label]["fog"]["aggregate"]
    x = d[label]["crosstalk"]["aggregate"]
    print(f"  [fog] zs {v['zs']:.3f} naive {v['naive']:.3f} oracle {v['oracle']:.3f}   "
          f"[crosstalk] zs {x['zs']:.3f} naive {x['naive']:.3f} oracle {x['oracle']:.3f}")
except Exception as e:
    print(f"  (summary parse failed: {e})")
EOF
  echo "" | tee -a "$LOG"
  sleep "$SLEEP"
done
