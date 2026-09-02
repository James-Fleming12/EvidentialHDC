#!/usr/bin/env bash
# setup_mink_uv.sh: create a SEPARATE uv venv capable of running the vendored
# MinkowskiNet models (modules/GeoID/models/*) for the
# DGLSS++-on-MinkowskiNet experiment.
#
# MinkowskiEngine 0.5.4 requires torch < 2.0 and must be COMPILED against the
# installed torch, so this env uses torch 1.13.x and is independent of the main
# repo's torch-2.x venv. Building ME from source needs the CUDA TOOLKIT (nvcc),
# gcc, and openblas on the server (a driver alone is not enough).
#
# Prerequisites (run first on the server):
#   nvidia-smi | head -4                     # driver / CUDA version
#   python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Then set CU to the torch wheel cuda tag that the driver supports:
#   CU=cu118  (needs driver >= 520)     CU=cu121  (needs driver >= 530)
#   CU=cu117  (needs driver >= 515)
#
# Usage (from the repo root on the server):
#   PY_VER=3.10 CU=cu118 bash setup_mink_uv.sh
#   source .venv-mink/bin/activate && python -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
set -euo pipefail

PY_VER="${PY_VER:-3.10}"
CU="${CU:-cu118}"
VENV="${VENV:-.venv-mink}"
ME_DIR="${ME_DIR:-$HOME/MinkowskiEngine}"

echo "=== MinkowskiNet uv setup (python $PY_VER, torch-cuda $CU, venv $VENV) ==="

# 1. venv
if [ ! -d "$VENV" ]; then
  echo "[1/4] creating uv venv (python $PY_VER)..."
  uv venv --python "$PY_VER" "$VENV"
else
  echo "[1/4] venv exists, reusing"
fi

# 2. torch 1.13 + the rest (ME needs torch<2.0)
echo "[2/4] installing torch==1.13.1 ($CU) + requirements_mink.txt..."
uv pip install --python "$VENV/bin/python" \
  --index-url "https://download.pytorch.org/whl/$CU" \
  "torch==1.13.1" "torchvision==0.14.1"
uv pip install --python "$VENV/bin/python" -r requirements_mink.txt

# 3. MinkowskiEngine 0.5.4 from source (compiles against the torch just installed)
echo "[3/4] building MinkowskiEngine 0.5.4 from source (needs CUDA toolkit + openblas)..."
if [ ! -d "$ME_DIR" ]; then
  git clone --recursive --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
fi
( cd "$ME_DIR" && \
  "$VENV/bin/python" setup.py install --blas=openblas 2>/dev/null || \
  "$VENV/bin/python" -m pip install -e . --install-option="--blas=openblas" -v --no-deps )

echo "[4/4] verifying..."
"$VENV/bin/python" -c "import torch, MinkowskiEngine as ME; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('MinkowskiEngine', ME.__version__)"
echo ""
echo "=== done. activate with: source $VENV/bin/activate ==="
echo "Next (a SEPARATE, larger task): the point-cloud data pipeline + porting"
echo "the DGLSS++ losses (GMSIFC/LSCC/SupCon) to the sparse MinkowskiNet, and"
echo "switching the GenTrainer AMP to fp32 for torch 1.13."
