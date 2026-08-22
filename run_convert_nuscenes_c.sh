#!/usr/bin/env bash
# run_convert_nuscenes_c.sh: validate the downloaded NuScenes-C, overlay it onto
# the base nuScenes dataroot, and convert to SemanticKITTI format (the same
# KittiConverter that produced /mnt/alpha/jmfleming/nuscenes_kitti).
#
# HOW NuScenes-C overlays the base (from the Robo3D generators): the corrupted
# archive contains ONLY the relative files
#     <cond>/<sev>/samples/LIDAR_TOP/*.pcd.bin      (corrupted scans)
#     <cond>/<sev>/lidarseg/v1.0-trainval/*.bin     (labels)
# with paths RELATIVE to the base nuScenes dataroot. There is no metadata in
# the archive -- the devkit resolves filenames from the BASE v1.0-trainval.
# So conversion = overlay corrupted files onto the base, then run the converter
# per condition/severity.
#
# Pipeline:
#   1. Validate: base nuScenes has v1.0-trainval metadata; each requested
#      <cond>/<sev> has samples/LIDAR_TOP + lidarseg/v1.0-trainval.
#   2. rsync the corrupted files OVER the base dataroot (a working copy, so the
#      pristine base is untouched).
#   3. KittiConverter.nuscenes_gt_to_semantickitti on the overlaid root ->
#      <out>/<cond>/<sev>/sequences/XXXX/{velodyne,labels,...}
#
# Usage:
#   bash run_convert_nuscenes_c.sh \
#       --nusc_c_root /mnt/bravo/jmfleming/OpenDataLab___nuScenes-C/raw/nuScenes-C \
#       --nusc_base /mnt/alpha/jmfleming/nuscenes \
#       --out_root /mnt/bravo/jmfleming/nuscenes_c_kitti
#   CONDS="fog,crosstalk" SEVS="heavy" bash run_convert_nuscenes_c.sh ...
#
# Prereqs: nuscenes-devkit in the venv (pip install nuscenes-devkit).

set -u
set -o pipefail

# --- args (positional or env) ---
NUSC_C_ROOT="${NUSC_C_ROOT:-/mnt/bravo/jmfleming/OpenDataLab___nuScenes-C/raw/nuScenes-C}"
NUSC_BASE="${NUSC_BASE:-/mnt/alpha/jmfleming/nuscenes}"
OUT_ROOT="${OUT_ROOT:-/mnt/bravo/jmfleming/nuscenes_c_kitti}"
CONDS="${CONDS:-fog,crosstalk,snow,wet_ground,incomplete_echo,beam_missing,motion_blur,cross_sensor}"
SEVS="${SEVS:-heavy,moderate,light}"
CONVERTER="${CONVERTER:-dataset/export_semantickitti.py}"
CONVERTER_DIR="$(cd "$(dirname "$CONVERTER")/.." 2>/dev/null && pwd || echo '')"

echo "NuScenes-C root: $NUSC_C_ROOT"
echo "Base nuScenes:   $NUSC_BASE"
echo "Output root:     $OUT_ROOT"
echo "Conditions:      $CONDS"
echo "Severities:      $SEVS"

fail() { echo "ERROR: $1" >&2; exit 1; }

# --- 1. validate ---
[ -d "$NUSC_C_ROOT" ] || fail "nusc_c_root '$NUSC_C_ROOT' does not exist"
[ -d "$NUSC_BASE/v1.0-trainval" ] || fail "base nuScenes '$NUSC_BASE' lacks v1.0-trainval metadata"

echo ""
echo "=== Validation ==="
cond_list=$(echo "$CONDS" | tr ',' ' ')
sev_list=$(echo "$SEVS" | tr ',' ' ')
missing=0
for c in $cond_list; do
  found=0
  for s in $sev_list; do
    d="$NUSC_C_ROOT/$c/$s"
    if [ -d "$d" ]; then
      nscan=$(ls "$d/samples/LIDAR_TOP/" 2>/dev/null | wc -l)
      nlbl=$(ls "$d/lidarseg/v1.0-trainval/" 2>/dev/null | wc -l)
      echo "  OK   $c/$s : $nscan scans, $nlbl labels"
      found=1
    else
      echo "  --   $c/$s : missing (skipping)"
    fi
  done
  [ $found -eq 1 ] || { echo "  !!   $c : no severity dir found"; missing=1; }
done
[ $missing -eq 0 ] || echo "WARNING: at least one condition has no severity dirs"

# --- 2. overlay onto a working copy of the base ---
echo ""
echo "=== Overlay corrupted files onto base (working copy) ==="
WORK_ROOT="$OUT_ROOT/_overlay"
for c in $cond_list; do
  for s in $sev_list; do
    d="$NUSC_C_ROOT/$c/$s"
    [ -d "$d" ] || continue
    # copy the base metadata once
    if [ ! -d "$WORK_ROOT/v1.0-trainval" ]; then
      echo "  copying base metadata -> $WORK_ROOT/v1.0-trainval"
      cp -r "$NUSC_BASE/v1.0-trainval" "$WORK_ROOT/"
    fi
    # the devkit also loads map masks (map.json -> maps/*.png)
    if [ ! -d "$WORK_ROOT/maps" ] && [ -d "$NUSC_BASE/maps" ]; then
      echo "  copying base maps -> $WORK_ROOT/maps"
      cp -r "$NUSC_BASE/maps" "$WORK_ROOT/"
    fi
    echo "  overlaying $c/$s -> $WORK_ROOT"
    mkdir -p "$WORK_ROOT/samples/LIDAR_TOP" "$WORK_ROOT/lidarseg/v1.0-trainval"
    cp -n "$d"/samples/LIDAR_TOP/* "$WORK_ROOT/samples/LIDAR_TOP/" 2>/dev/null || true
    cp -n "$d"/lidarseg/v1.0-trainval/* "$WORK_ROOT/lidarseg/v1.0-trainval/" 2>/dev/null || true
  done
done

# --- 3. convert per condition/severity ---
echo ""
echo "=== Conversion (per condition/severity) ==="
rc=0
for c in $cond_list; do
  for s in $sev_list; do
    d="$NUSC_C_ROOT/$c/$s"
    [ -d "$d" ] || continue
    # fresh overlay for this (cond,sev) so the converter sees ONLY this corruption
    rm -rf "$WORK_ROOT/samples" "$WORK_ROOT/lidarseg"
    mkdir -p "$WORK_ROOT/samples/LIDAR_TOP" "$WORK_ROOT/lidarseg/v1.0-trainval"
    cp "$d"/samples/LIDAR_TOP/* "$WORK_ROOT/samples/LIDAR_TOP/" 2>/dev/null || true
    cp "$d"/lidarseg/v1.0-trainval/* "$WORK_ROOT/lidarseg/v1.0-trainval/" 2>/dev/null || true
    # The base lidarseg.json covers all 34149 trainval frames, but the overlay
    # only provides the corrupted val labels. The devkit asserts
    # num_lidarseg_recs == num_label_files at init, so prune lidarseg.json to
    # the records whose label files are actually present here.
    python - "$WORK_ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
label_dir = os.path.join(root, 'lidarseg', 'v1.0-trainval')
existing = {os.path.basename(p) for p in glob.glob(os.path.join(label_dir, '*.bin'))}
lj_path = os.path.join(root, 'v1.0-trainval', 'lidarseg.json')
with open(lj_path) as f:
    recs = json.load(f)
recs = [r for r in recs if os.path.basename(r['filename']) in existing]
with open(lj_path, 'w') as f:
    json.dump(recs, f)
print(f"  pruned lidarseg.json -> {len(recs)} records ({len(existing)} label files)")
PY
    out="$OUT_ROOT/$c/$s"
    mkdir -p "$out"
    echo "--- $c/$s -> $out ---"
    if [ -n "$CONVERTER_DIR" ]; then
      ( cd "$CONVERTER_DIR" && \
        python "$CONVERTER" nuscenes_gt_to_semantickitti \
          --nusc_dir "$WORK_ROOT" \
          --nusc_skitti_dir "$out" \
          --nusc_version v1.0-trainval )
    else
      python "$CONVERTER" nuscenes_gt_to_semantickitti \
        --nusc_dir "$WORK_ROOT" --nusc_skitti_dir "$out" --nusc_version v1.0-trainval
    fi || { echo "  !! conversion failed for $c/$s"; rc=1; }
  done
done
rm -rf "$WORK_ROOT"

echo ""
if [ $rc -eq 0 ]; then
  echo "=== CONVERSION COMPLETE ==="
  echo "KITTI-format NuScenes-C at: $OUT_ROOT"
  echo "  $OUT_ROOT/<condition>/<severity>/sequences/XXXX/{velodyne,labels,calib.txt,poses.txt,times.txt}"
  echo "Point the full-dataset diag's parser root at a condition/severity dir, e.g."
  echo "  $OUT_ROOT/fog/heavy  (with config/labels/nuscenes_new.yaml)"
else
  echo "=== CONVERSION PARTIAL (some conditions failed) ==="
  exit 1
fi
