#!/usr/bin/env bash
# run_download_nuscenes_c.sh: download NuScenes-C to /mnt/bravo/jmfleming via the
# OpenDataLab CLI (the same source/layout as the existing SemanticKITTI-C:
# /mnt/bravo/jmfleming/OpenDataLab___SemanticKITTI-C).
#
# Usage:
#   bash run_download_nuscenes_c.sh
#
# Prereqs:
#   - pip install opendatalab            (the 'odl' CLI)
#   - odl login                          (OpenDataLab account; paste the token)
#   - enough disk on /mnt/bravo (NuScenes-C is ~200-300GB with labels)

set -u
set -o pipefail

DST="/mnt/bravo/jmfleming"
DATASET="nuScenes-C"          # OpenDataLab dataset id (case-sensitive, may be 'nuscenes-c')
mkdir -p "$DST"

if ! command -v odl >/dev/null 2>&1; then
  echo "ERROR: 'odl' not found. Install with: pip install opendatalab"
  exit 1
fi

echo "=== Downloading $DATASET to $DST (OpenDataLab CLI) ==="
cd "$DST"

# Search first so you can confirm the exact id before committing GBs:
echo "--- available matches on OpenDataLab ---"
odl search "$DATASET" || echo "(search not supported by this odl version; proceeding with '$DATASET')"

# Download. This creates $DST/OpenDataLab___<dataset>/raw/<dataset>/
odl get "$DATASET" || {
  echo ""
  echo "If 'odl get $DATASET' failed, list ids with: odl ls"
  echo "then rerun with the exact id, e.g.: odl get OpenDataLab___nuScenes-C"
  exit 1
}

echo ""
echo "=== Download complete ==="
echo "Expected layout (mirrors SemanticKITTI-C):"
echo "  $DST/OpenDataLab___nuScenes-C/raw/nuScenes-C/"
echo "    v1.0-trainval/"
echo "    <condition>/heavy|moderate|light/lidarseg/... + sample/LIDAR_TOP/..."
echo ""
echo "Point the full-dataset diag's --kittic_dir-style root at the extracted"
echo "corrupted dir (e.g. $DST/OpenDataLab___nuScenes-C/raw/nuScenes-C) when"
echo "evaluating the NuScenes-trained cov-shift extractor on NuScenes-C."
