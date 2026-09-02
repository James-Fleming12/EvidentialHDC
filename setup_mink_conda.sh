#!/usr/bin/env bash
# setup_mink_conda.sh: a conda env capable of building/running the vendored
# MinkowskiNet models (modules/GeoID/models/*) for the
# DGLSS++-on-MinkowskiNet experiment.
#
# WHY conda: the server has torch 2.13+cu130 and nvcc 13.2. MinkowskiEngine
# 0.5.4 (2022) needs torch < 2.0 AND a CUDA 11.x toolchain to compile. Conda
# installs the CUDA 11.8 toolkit (with nvcc) alongside python 3.10 and a
# torch 1.13.1+cu118 wheel, all isolated from the main torch-2.x env. The
# 595 driver is backward compatible, so cu118 binaries run on it.
#
# Usage (on the server, from the repo root):
#   bash setup_mink_conda.sh
#   conda activate mink && python -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
#
# If the ME build fails on a newer gcc, retry with:
#   ME_MAX_JOBS=8 CXXFLAGS="-Wno-unused-but-set-variable" bash setup_mink_conda.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-mink}"
PY_VER="${PY_VER:-3.10}"
CU_TAG="${CU_TAG:-cu117}"          # torch wheel tag: 1.13.1's newest CUDA build is cu117
                                   # (cu118 wheels start at torch 2.0; GeoID's setup also
                                   # resolves to cu117 from default PyPI)
CUDA_TK="${CUDA_TK:-11.8}"         # conda CUDA toolkit for nvcc (11.8 compiles ME 0.5.4;
                                   # minor-version mix with torch cu117 is fine)
ME_DIR="${ME_DIR:-$HOME/MinkowskiEngine}"

echo "=== MinkowskiNet conda setup ($ENV_NAME, py $PY_VER, torch $CU_TAG, cuda-toolkit $CUDA_TK) ==="

# 1. create the env
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "[1/4] creating conda env $ENV_NAME (python $PY_VER)..."
  conda create -n "$ENV_NAME" python="$PY_VER" -y
else
  echo "[1/4] conda env $ENV_NAME exists"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2. CUDA 11.x compiler + headers (nvcc) from the nvidia channel.
# NOTE: use the MINIMAL split packages (cuda-nvcc, cuda-cudart-dev), NOT the
# cuda-toolkit metapackage -- that pulls nsight-compute / nsight-systems etc.
# (huge, not cached, and fails with "no package in cache directories").
echo "[2/4] installing cuda-nvcc / cuda-cudart-dev (provides nvcc + headers)..."
conda install -n "$ENV_NAME" -c nvidia "cuda-nvcc=$CUDA_TK" "cuda-cudart-dev=$CUDA_TK" "cuda-cudart=$CUDA_TK" -y
conda activate "$ENV_NAME"
which nvcc && nvcc --version | tail -1

# 3. torch 1.13 + the rest (ME needs torch<2.0; use the cu117 build)
echo "[3/4] installing torch==1.13.1 ($CU_TAG) + requirements_mink.txt..."
pip install --index-url "https://download.pytorch.org/whl/$CU_TAG" \
  "torch==1.13.1+$CU_TAG" "torchvision==0.14.1+$CU_TAG"
pip install -r requirements_mink.txt

# 4. MinkowskiEngine 0.5.4 from source (compiles against the conda nvcc + torch)
echo "[4/4] building MinkowskiEngine 0.5.4 from source (verbose)..."
if [ ! -d "$ME_DIR" ]; then
  git clone --recursive --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git "$ME_DIR"
fi
export CUDA_HOME="$CONDA_PREFIX"   # point ME's build at the conda CUDA 11.8 toolkit
# NOTE: pip's --install-option was removed in modern pip, and `setup.py install`
# needs to run verbosely to surface compile errors. Build manually:
#   cd ~/MinkowskiEngine
#   python setup.py build_ext --blas=openblas
#   python setup.py install  --blas=openblas
# (the script no longer auto-builds -- run step 4 by hand so errors are visible)
echo "  build manually (see above); ME_DIR=$ME_DIR"

echo "verifying..."
python -c "import torch, MinkowskiEngine as ME; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('MinkowskiEngine', ME.__version__)"
echo ""
echo "=== done. activate with: conda activate $ENV_NAME ==="
echo "Next (a SEPARATE, larger task): the point-cloud data pipeline + porting"
echo "the DGLSS++ losses (GMSIFC/LSCC/SupCon) to the sparse MinkowskiNet, and"
echo "switching the GenTrainer AMP to fp32 for torch 1.13."
