#!/usr/bin/env bash
# Mid-training monitor for the 10h dircons medium run. While run_dircons_medium.sh
# trains, this snapshots the rolling checkpoint (SENet is overwritten every epoch)
# and runs the extractor_diff per-branch gate on each snapshot, so we can see whether
# the decoupling mechanism (corr dir_retention < 1, inv feat_cos high, oracle up) is
# developing DURING training -- and whether dir_w / res_w / lscc_corr need a
# reweighting before the full 10h is spent.
#
# Usage (in a SECOND terminal, while the medium run trains):
#   bash monitor_dircons.sh            # GPU 3 (uses an interleaved port of the run's GPU)
#
# Decision rules to watch (from logs/dircons_med_monitor.log, the per-snapshot table):
#   - corr_dir trending DOWN toward <1 on sig13/veg16 by mid-training  -> dircons works
#   - corr_dir stuck ~0.9 everywhere at epoch ~10+  -> dir_w too weak (0.1 -> 0.2),
#     or res_w too strong (0.05 -> 0.02); kill + relaunch
#   - inv_feat_cos dropping (unanchoring)           -> L_res too weak / LSCC missing;
#     raise res_w or restore lscc_corr
#   - corr_tightness collapsing                     -> L_dir fighting CE; lower dir_w
#   - oracle flat / TTA gap negative                -> stop, reweight, relaunch

set -u
GPU="${1:-3}"
METHOD="supcon_vib_dglsspp_corsupcon_residual_128_128_dircons"
MED_DIR="robust_diagnostic/logs/med_dircons"
CKPT="$MED_DIR/$METHOD/SENet"
REF_METHOD="supcon_vib_dglsspp_corsupcon"
REF_PATH="robust_diagnostic/logs/micro_corsupcon/$REF_METHOD"
SNAP_DIR="robust_diagnostic/logs/monitor_dircons"
SLEEP="${2:-2400}"   # seconds between snapshots (default 40 min)
mkdir -p "$SNAP_DIR"
LOG="logs/dircons_med_monitor.log"
: > "$LOG"
echo "Monitoring $CKPT every ${SLEEP}s; results -> $LOG" | tee -a "$LOG"

i=0
while [ ! -f "$CKPT" ]; do
  echo "[$(date +%H:%M)] checkpoint not present yet, waiting..." | tee -a "$LOG"
  sleep 600
done

last_ep=-1
while true; do
  i=$((i + 1))
  # read current epoch from the checkpoint header
  ep=$(python3 -c "
import torch
try:
    w = torch.load('$CKPT', map_location='cpu')
    print(w['epoch'])
except Exception:
    print('?')" 2>/dev/null)
  if [ "$ep" = "$last_ep" ]; then
    echo "[$(date +%H:%M)] epoch $ep unchanged (training done or stalled); exiting" | tee -a "$LOG"
    exit 0
  fi
  last_ep="$ep"

  # snapshot the rolling checkpoint (epoch-numbered copy so the run's SENet is untouched)
  snap="$SNAP_DIR/snap_ep${ep}"
  mkdir -p "$snap"
  cp "$CKPT" "$snap/SENet"
  echo "=== [$(date +%H:%M)] snapshot epoch $ep ===" | tee -a "$LOG"

  CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/extractor_diff_diag.py \
    --path_a "$REF_PATH" --method_a "$REF_METHOD" --label_a "corsupcon_micro" \
    --path_b "$snap" --method_b "$METHOD" --label_b "dircons_ep${ep}" \
    --inv_ch 128 --out "$SNAP_DIR/decouple2_gate_ep${ep}.json" \
    2>&1 | tee -a "$LOG" || echo "  (extractor_diff on ep $ep failed - checkpoint mid-write?)" | tee -a "$LOG"

  # per-snapshot one-line summary (car/ts/veg, fog): corr_dir + oracle
  python3 - "$SNAP_DIR/decouple2_gate_ep${ep}.json" "$METHOD" <<'EOF' 2>/dev/null | tee -a "$LOG"
import json, sys
fn, method = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(fn))
    v = d[method]["fog"]["per_class"]
    row = lambda c: f"{v[c]['oracle_iou']:.3f}/{v[c].get('corr_dir_retention', float('nan')):.2f}/{v[c].get('inv_feat_cos', float('nan')):.2f}"
    print(f"  [fog] agg oracle {d[method]['fog']['aggregate']['oracle']:.4f}  "
          f"car or/corr_dir/inv_fc {row('4')}  sig13 {row('13')}  veg16 {row('16')}")
except Exception as e:
    print(f"  (summary parse failed: {e})")
EOF

  echo "" | tee -a "$LOG"
  sleep "$SLEEP"
done
