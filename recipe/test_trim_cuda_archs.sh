#!/usr/bin/env bash
set -e

# trim_cuda_archs.py bundles its own unit test suite; run it as an install-time check
python -m unittest cf_nvidia_tools.trim_cuda_archs -v

# exercise the documented bash usage pattern end-to-end
export CUDAARCHS="75-real;80-real;90a-real"
export CUDAARCHS="$(trim-cuda-archs CUDAARCHS 8.0)"
[ "$CUDAARCHS" = "80-real;90a-real" ]

# bad arch exits 2
[ $(trim-cuda-archs CUDAARCHS not-an-arch &>/dev/null; echo $?) -eq 2 ]

# check_cuda_arch.py bundles its own unit test suite; run it as an install-time check
python -m unittest cf_nvidia_tools.check_cuda_arch -v

# check_cuda_arch.py's own argument validation
[ $(check-cuda-arch &>/dev/null; echo $?) -eq 2 ]

# sanity check against a real redistributable library: libcublas (CUDA 13) still ships
# SASS for sm_75, so the computed lowest common CUDA architecture must be 7.5
cuda_arch_version=7.5 check-cuda-arch "${CONDA_PREFIX}"/lib/libcublas.so.*
