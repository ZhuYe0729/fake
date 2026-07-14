#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
OUT_DIR="${ROOT}/max_speed/prefill_decode/results/speed"
# The shared benchmark_one.py leaves Llama3.1 at its 131072-token config limit,
# which cannot reserve KV cache on this GPU. This runner sets the workload limit
# to 2128 through our phase benchmark's HF override while retaining one fresh
# vLLM process for every timed sample.
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${ROOT}/../llama2-7b-chat/scripts/benchmark_phase_hetero.py" \
  --checkpoint "${ROOT}/max_speed/prefill_decode/checkpoint" --scenario prefill_decode \
  --output-dir "${OUT_DIR}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --warmup-iters 1 --iters 10
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" --root "${ROOT}"
