#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-0}"
BENCH_GPUS="${BENCH_GPUS:-0,1,2,3,4,5}"
QUALITY_GPUS="${QUALITY_GPUS:-0,1,2}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/root/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export SPARSE_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_WRAPPER}"
export HF_HOME="${HF_HOME:-/home/agent/wja/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/agent/wja/.cache/huggingface/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"${PYTHON}" "${ROOT}/scripts/export_selected8_vllm_checkpoints.py" --force
"${PYTHON}" "${ROOT}/scripts/benchmark_selected8_vllm_parallel.py" \
  --python "${PYTHON}" \
  --gpus "${BENCH_GPUS}"
"${PYTHON}" "${ROOT}/scripts/eval_selected8_quality_vllm_parallel.py" \
  --python "${PYTHON}" \
  --gpus "${QUALITY_GPUS}"
"${PYTHON}" "${ROOT}/scripts/summarize_selected8_vllm.py"
