#!/usr/bin/env bash
# setup_nuscenes_c_env.sh: brand-new conda env JUST for the NuScenes-C
# download + conversion pipeline (nothing else, so nothing breaks).
#
# Python 3.11 is the key choice: it has prebuilt wheels for every dependency
# (shapely bundles GEOS, oss2/crcmod for the OpenDataLab CLI), avoiding the
# Python-3.14 source-build failure entirely.
#
# Usage (on the server):
#   bash setup_nuscenes_c_env.sh
#   conda activate nusc_convert

set -u
set -o pipefail

ENV_NAME="${ENV_NAME:-nusc_convert}"
PYTHON="${PYTHON:-3.11}"

echo "=== Creating conda env '$ENV_NAME' (python $PYTHON) ==="
conda create -y -n "$ENV_NAME" "python=$PYTHON" || exit 1

echo ""
echo "=== Installing NuScenes-C pipeline deps ==="
# `source activate` works in scripts; conda envs have their own pip.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# nuscenes-devkit: the KittiConverter (export_semantickitti.py) imports
#   nuscenes, fire, matplotlib, numpy, PIL, pyquaternion
pip install --upgrade pip
pip install nuscenes-devkit opendatalab

echo ""
echo "=== Verification ==="
python - <<'EOF'
import sys
print("python:", sys.version.split()[0])
mods = ["nuscenes", "fire", "shapely", "oss2", "crcmod", "matplotlib", "PIL", "pyquaternion", "numpy"]
ok = True
for m in mods:
    try:
        __import__(m)
        print(f"  {m:16s} OK")
    except Exception as e:
        ok = False
        print(f"  {m:16s} FAIL: {e}")
if not ok:
    print("SOME IMPORTS FAILED")
    sys.exit(1)
print("All imports OK -- env ready for download + conversion.")
EOF

echo ""
echo "Next steps:"
echo "  1. conda activate $ENV_NAME"
echo "  2. odl login"
echo "  3. bash ~/EvidentialHDC/run_download_nuscenes_c.sh"
echo "  4. NUSC_C_ROOT=... NUSC_BASE=... OUT_ROOT=... bash ~/EvidentialHDC/run_convert_nuscenes_c.sh"
