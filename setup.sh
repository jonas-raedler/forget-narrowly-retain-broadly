#!/usr/bin/env bash
# Set up the `unlearn` conda environment (ordered dependencies).
#
# Tested with CUDA 12.8.
#
# Building the CUDA extensions below (flash-attn, causal-conv1d) needs a modern C++ compiler
# (GCC >= 10) on PATH. Make sure one is available before running, e.g.:
#   - RHEL / CentOS (Software Collections):  source scl_source enable gcc-toolset-11
#   - Ubuntu / Debian:                       sudo apt install build-essential g++-11
#   - any platform via conda:                conda install -c conda-forge gxx_linux-64=11
#
# Usage:
#   bash setup.sh
#   conda activate unlearn
#
# ENV_NAME overrides the environment name (default: unlearn):
#   ENV_NAME=my_env bash setup.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-unlearn}"

# 1) Create the env and activate it for the rest of this script
conda create --name "${ENV_NAME}" python=3.12 -y
# `conda activate` needs the shell hook when running non-interactively
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# 2) NumPy + PyTorch (CUDA 12.8 build)
pip install "numpy<=2"
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# 3) Transformers
pip install transformers==5.3.0

# 4) Kernels / attention  (make sure your CUDA is 12.8 before building)
# FORCE_BUILD compiles the CUDA kernels locally instead of downloading a prebuilt wheel,
# so the extension links against this machine's glibc. If your machine's glibc matches the
# published wheels, you can drop the FORCE_BUILD vars to install the prebuilt wheel instead.
# MAX_JOBS caps the parallel compile jobs so the build does not run out of memory.
MAX_JOBS="${MAX_JOBS:-4}"
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" pip install causal-conv1d==1.6.0 --no-build-isolation
pip install flash-linear-attention==0.4.1
pip install psutil
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" pip install flash-attn==2.8.3 --no-build-isolation

# 5) Remaining dependencies
conda env update -n "${ENV_NAME}" --file environment.yml

echo
echo "Done. Activate the environment with:  conda activate ${ENV_NAME}"