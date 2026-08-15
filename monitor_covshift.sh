#!/usr/bin/env bash
# Mid-training monitor for the covariate-shift medium run (inputin_in_chan).
# While run_covshift_medium.sh trains, this snapshots the rolling checkpoint
# (SENet is overwritten every epoch) and runs the extractor_diff gate on each
# snapshot vs plain DGLSS++, logging the oracle + naive per epoch so we can find
# the OPTIMAL epoch window -- critical because the family degrades past 21 ep
# (Iteration 8.1) and this extractor may peak earlier or later.
#
# EVERYTHING is written to ONE consolidated log: logs/covshift_med_monitor.log
# (epoch reads, stall checks, full extractor_diff output, and the per-snapshot
# summary line). Snapshot checkpoints and gate JSONs live under
# robust_diagnostic/logs/monitor_covshift/.
#
# Usage (in a SECOND terminal, while the medium run trains):
#   bash monitor_covshift.sh            # GPU 3, every 30 min
#   bash monitor_covshift.sh 3 1800     # custom interval (seconds)
#   bash monitor_covshift.sh 3 1800 60  # also custom stall threshold (minutes)
#
# Decision rules (from logs/covshift_med_monitor.log):
#   - oracle trending UP on fog AND crosstalk through mid-training -> on track
#   - oracle PEAKS then FALLS (watch for the knee) -> note the best epoch; after
#     training, the best checkpoint can be resumed-to / used for the battery
#   - naive TTA also UP (it should, from the micro: crosstalk 0.29 vs robust 0.13)
#   - if oracle flat or below plain DGLSS++ at epoch ~15+, the cov-shift win does
#     not transfer to medium -> stop, do not commit further

set -u
GPU="${1:-3}"
SLEEP="${2:-1800}"
STALL_MIN="${3:-60}"
METHOD="supcon_vib_dglsspp_inputin_in_chan"
MED_DIR="robust_diagnostic/logs/med_$METHOD"
CKPT="$MED_DIR/$METHOD/SENet"
DGLSSPP_PATH="robust_diagnostic/logs/supcon_vib_dglsspp"
DGLSSPP_METHOD="supcon_vib_dglsspp"
SNAP_DIR="robust_diagnostic/logs/monitor_covshift"
mkdir -p "$SNAP_DIR"
LOG="logs/covshift_med_monitor.log"
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

# Robust epoch read: torch.load fails if the checkpoint is mid-write; retry a few
# times with a short gap before giving up (so the snapshot label is the true epoch).
read_epoch() {
  for attempt in 1 2 3 4 5; do
    ep=$(python3 -c "
import torch
try:
    print(torch.load('$CKPT', map_location='cpu')['epoch'])
except Exception:
    print('')" 2>/dev/null)
    if [ -n "$ep" ] && [ "$ep" != "" ]; then
      echo "$ep"; return 0
    fi
    sleep 2
  done
  echo ""
}

log "Monitoring $CKPT every ${SLEEP}s + analysis time; consolidated log -> $LOG"
log "Stall detection: exit only if the checkpoint has not been written for >${STALL_MIN} min"

while [ ! -f "$CKPT" ]; do
  log "[$(date +%H:%M)] checkpoint not present yet, waiting..."
  sleep 600
done

last_ep=""
while true; do
  # NOTE: interval between reads = SLEEP + analysis time, and the analysis shares the
  # GPU (slowing training), so a repeated epoch number is NOT a stall. Stall is judged
  # by the checkpoint FILE mtime: training overwrites SENet every epoch.
  ep=$(read_epoch)

  age_s=$(python3 -c "
import os, time
p = '$CKPT'
if os.path.exists(p):
    print(int(time.time() - os.path.getmtime(p)))
else:
    print(999999)" 2>/dev/null)
  age_min=$(( ${age_s:-999999} / 60 ))

  if [ -z "$ep" ]; then
    log "[$(date +%H:%M)] could not read epoch (checkpoint mid-write); treating as epoch $last_ep"
    ep="$last_ep"
  fi

  if [ -n "$last_ep" ] && [ "$ep" = "$last_ep" ]; then
    if [ "$age_min" -gt "$STALL_MIN" ]; then
      log "[$(date +%H:%M)] epoch $ep unchanged AND checkpoint unwritten for ${age_min} min "
          "(>${STALL_MIN}); training done or stuck -- exiting"
      exit 0
    else
      log "[$(date +%H:%M)] epoch $ep still current, checkpoint written ${age_min} min ago "
          "-- training slowed by analysis, continuing..."
    fi
  fi
  last_ep="$ep"

  snap="$SNAP_DIR/snap_ep${ep}"
  mkdir -p "$snap"
  cp "$CKPT" "$snap/SENet"
  log "=== [$(date +%H:%M)] snapshot epoch $ep ==="

  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$DGLSSPP_PATH" --method_a "$DGLSSPP_METHOD" --label_a "dglsspp_med" \
    --path_b "$snap" --method_b "$METHOD" --label_b "covshift_ep${ep}" \
    --frames 50 --pool_size 50000 --val_size 50000 \
    --out "$SNAP_DIR/gate_ep${ep}.json" \
    2>&1 | tee -a "$LOG" || log "  (extractor_diff on ep $ep failed - checkpoint mid-write?)"

  python3 - "$SNAP_DIR/gate_ep${ep}.json" "covshift_ep${ep}" <<'EOF' 2>/dev/null | tee -a "$LOG"
import json, sys
fn, label = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(fn))
    v = d[label]["fog"]["aggregate"]
    x = d[label]["crosstalk"]["aggregate"]
    print(f"  >>> SUMMARY ep {label}  [fog] zs {v['zs']:.3f} naive {v['naive']:.3f} "
          f"oracle {v['oracle']:.3f}  [crosstalk] zs {x['zs']:.3f} naive {x['naive']:.3f} "
          f"oracle {x['oracle']:.3f}")
except Exception as e:
    print(f"  (summary parse failed: {e})")
EOF
  log ""
  sleep "$SLEEP"
done
