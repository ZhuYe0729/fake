#!/usr/bin/env bash
set -euo pipefail

# conda activate vllm
source /root/workspaces/cospaq/cutlass_wrapper/scripts/activate_cutlass_wrapper.sh

export CUDA_VISIBLE_DEVICES=0
export REPO=/root/workspaces/cospaq/fake
export VLLM_ROOT=/root/workspaces/cospaq/vllm-cospaq/vllm
export MODEL=/root/data/model/Llama-2-7b-chat-hf
export CUTLASS_ROOT=/root/workspaces/cospaq/cutlass_wrapper
export VLLM_PYTHON=/root/miniconda3/envs/vllm/bin/python
export COSPAQ_PYTHON=/root/miniconda3/envs/cospaq/bin/python

export HF_HOME=/root/data/huggingface
export HF_DATASETS_CACHE=/root/data/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/benchmark_vllm_scenarios.py" \
  --methods dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \
  --scenarios prefill_only,prefill_decode \
  "$@"
