#!/usr/bin/env bash
# run_al_nuscenes_c.sh: evaluate extractors on NuScenes-C (KITTI-format, from
# run_convert_nuscenes_c.sh) with the SAME full-dataset machinery as
# run_al_full_dataset.sh:
#   * zero-shot W0  : exact-ridge fit on <= clean_fit_n CLEAN KITTI seq-08 points
#   * ceiling  W*   : exact-ridge fit on <= pool_cap corrupted-pool points
#   * AL W_res      : W0 + U8 C on the 56+500 random bank (oracle U)
#   * VAL           : ALL points of ALL frames of the condition/severity
# per condition/severity under --nusc_c_dir, keyed at
#   extractors[<label>]['nuscenes_c']['<cond>/<sev>'].
#
# Requires --nusc_labels to point at a yaml whose `split.valid` lists the scene
# indices ACTUALLY present in the converted NuScenes-C output (the C archive
# covers the official nuScenes val split, which differs from nuscenes_new.yaml's
# valid list). Generate it from the converted sequences dirs first.
#
# Usage:
#   bash run_al_nuscenes_c.sh 3                         # nusc_cov extractor, all conds x heavy
#   NUSC_C_KITTI=/path/to/nuscenes_c_kitti bash run_al_nuscenes_c.sh 3
#   CONDS=fog,crosstalk SEVS=heavy,moderate bash run_al_nuscenes_c.sh 3
#   EXTRACTORS="..." NUSC=0 bash run_al_nuscenes_c.sh 3   # skip pristine NuScenes ref
#
# Output: robust_diagnostic/logs/al_nuscenes_c.json

set -u
set -o pipefail
GPU="${1:-3}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
SEVS="${SEVS:-heavy}"
NUSC="${NUSC:-0}"           # 0 = only NuScenes-C (pristine ref optional)
BAL="${BAL:-0}"             # 0 = skip the (dead-end) class-balanced probes
NUSC_C_KITTI="${NUSC_C_KITTI:-/mnt/bravo/jmfleming/nuscenes_c_kitti}"
NUSC_C_LABELS="${NUSC_C_LABELS:-config/labels/nuscenes_c.yaml}"
EXTRACTORS="${EXTRACTORS:-nusc_cov:supcon_vib_dglsspp_inputin_in_chan:robust_diagnostic/logs/nusc_covshift_21ep}"
echo "Using GPU $GPU (conds=$CONDS, sevs=$SEVS, nusc=$NUSC, bal=$BAL)"
echo "NuScenes-C KITTI root: $NUSC_C_KITTI  (labels: $NUSC_C_LABELS)"

[ -f "$NUSC_C_LABELS" ] || { echo "ERROR: $NUSC_C_LABELS missing (see run_al_nuscenes_c.sh header)"; exit 1; }

eval CUDA_VISIBLE_DEVICES=$GPU uv run python robust_diagnostic/al_full_dataset_diag.py \
  --label "nuscenes_c" \
  --conds none --nusc "$NUSC" --bal "$BAL" \
  --extractors "$EXTRACTORS" \
  --nusc_dir /mnt/alpha/jmfleming/nuscenes_kitti \
  --nusc_labels "$NUSC_C_LABELS" \
  --nusc_c_dir "$NUSC_C_KITTI" \
  --nusc_c_conds "$CONDS" \
  --nusc_c_sevs "$SEVS" \
  --out "robust_diagnostic/logs/al_nuscenes_c.json" \
  2>&1 | tee "logs/al_nuscenes_c.log"

RC=$?
if [ $RC -eq 0 ]; then
  echo "=== NUSCENES-C OK ==="
  echo "Check robust_diagnostic/logs/al_nuscenes_c.json:"
  echo "  extractors[<label>].nuscenes_c['<cond>/<sev>'] = { linear_frozen, linear_ceiling, ... }"
else
  echo "=== NUSCENES-C FAILED (exit $RC) ==="
  exit $RC
fi
