#!/usr/bin/env bash
# Export one legal canonical policy and measure five continuous phase-E2E runs.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 POLICY_ID PHYSICAL_GPU" >&2; exit 2; fi
POLICY_ID="$1"; GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/root/wja/project/my/cospaq/fake"
MODEL="/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORT="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
POLICY="${ROOT}/llama2_7b_chat/speed/calibration/policies/${POLICY_ID}.json"
CHECKPOINT="${ROOT}/llama2_7b_chat/speed/calibration/checkpoints/${POLICY_ID}"
RUNS="${ROOT}/llama2_7b_chat/speed/calibration/runs/${POLICY_ID}"
mkdir -p "${RUNS}" "${ROOT}/logs/speed"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORT}" --policy-json "${POLICY}" --model-path "${MODEL}" --output-dir "${CHECKPOINT}" --force \
    > "${ROOT}/logs/speed/export_${POLICY_ID}.log" 2>&1
fi
if [ ! -s "${RUNS}/continuous_summary.json" ]; then
  export TORCH_CUDA_ARCH_LIST=12.0
  export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
  export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" --scenario prefill_decode --output-dir "${RUNS}" \
    --gpu-memory-utilization 0.85 --warmup-iters 6 --iters 5 --single-phase main --single-output "${RUNS}/continuous_summary.json" \
    --single-samples-dir "${RUNS}" --reuse-llm > "${ROOT}/logs/speed/measure_${POLICY_ID}.log" 2>&1
fi
