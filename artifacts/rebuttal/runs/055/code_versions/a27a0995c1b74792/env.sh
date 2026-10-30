#!/usr/bin/env bash
# Source this file before run055 commands. GPU commands need host execution.
export DASO_REPO_ROOT=/root/wja2/project/my/fake
export DASO_RUN_ROOT="$DASO_REPO_ROOT/artifacts/rebuttal/runs/055"
export DASO_MODEL_PATH=/root/wja2/data/models/Qwen/Qwen3-32B
export DASO_VLLM_ROOT="$DASO_RUN_ROOT/dependencies/phase-vllm"
export DASO_CUTLASS_WRAPPER_ROOT="$DASO_REPO_ROOT/fake/kernels/cutlass/cutlass_wrapper"
export PYTHONPATH="$DASO_RUN_ROOT/code:$DASO_VLLM_ROOT/vllm:$DASO_VLLM_ROOT:$DASO_CUTLASS_WRAPPER_ROOT:$DASO_REPO_ROOT/daso/src"
export PATH="$DASO_RUN_ROOT/.venv/bin:$PATH"
export CUDA_HOME="$DASO_RUN_ROOT/dependencies/cuda128"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++ CC=/usr/bin/gcc
export CPATH="$CUDA_HOME/targets/x86_64-linux/include"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib"
for daso_cuda_component in cuda_runtime cublas cusparse cusolver curand; do
  export CPATH="$CPATH:$DASO_RUN_ROOT/.venv/lib/python3.12/site-packages/nvidia/$daso_cuda_component/include"
  export LIBRARY_PATH="$LIBRARY_PATH:$DASO_RUN_ROOT/.venv/lib/python3.12/site-packages/nvidia/$daso_cuda_component/lib"
done
unset daso_cuda_component
export HF_HOME="$DASO_RUN_ROOT/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$DASO_RUN_ROOT/extensions/torch"
export TRITON_CACHE_DIR="$DASO_RUN_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$DASO_RUN_ROOT/cache/vllm"
export MAX_JOBS=8 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR="$DASO_RUN_ROOT/extensions/nvfp4"
export CUTLASS_WRAPPER_SPARSE_NVFP4_EXT_BUILD_DIR="$DASO_RUN_ROOT/extensions/sparse_nvfp4"
export CUTLASS_WRAPPER_SPARSE_BF16_EXT_BUILD_DIR="$DASO_RUN_ROOT/extensions/sparse_bf16"
export CUTLASS_WRAPPER_MARLIN_NVFP4_EXT_BUILD_DIR="$DASO_RUN_ROOT/extensions/marlin_nvfp4"
export MARLIN_INT4_EXT_BUILD_DIR="$DASO_RUN_ROOT/extensions/marlin_int4"
