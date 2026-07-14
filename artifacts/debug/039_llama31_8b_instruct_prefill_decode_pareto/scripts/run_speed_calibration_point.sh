#!/usr/bin/env bash
# Export one fixed policy and collect five fresh continuous phase-E2E samples.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 POLICY_ID PHYSICAL_GPU" >&2; exit 2; fi
POLICY_ID="$1"; GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/home/agent/wja/project/my/cospaq/fake"
MODEL="/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORT="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
RUN_GROUP="${RUN_GROUP:-speed_calibration_continuous085_w6}"
POLICY_GROUP="${POLICY_GROUP:-speed_calibration_util085}"
CHECKPOINT_GROUP="${CHECKPOINT_GROUP:-${RUN_GROUP}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LEGAL_POLICY="${ROOT}/${POLICY_GROUP}/policies/${POLICY_ID}.json"
if [ ! -f "${LEGAL_POLICY}" ]; then
  "${EXPORT_PYTHON}" "${ROOT}/scripts/make_legal_speed_policy.py" "${POLICY_ID}" > "${ROOT}/logs/legal_policy_${POLICY_ID}.log" 2>&1
fi
POLICY="${LEGAL_POLICY}"
CHECKPOINT="${ROOT}/${CHECKPOINT_GROUP}/checkpoints/${POLICY_ID}"
RUNS="${ROOT}/${RUN_GROUP}/runs/${POLICY_ID}"
mkdir -p "${RUNS}" "${ROOT}/logs"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORT}" --policy-json "${POLICY}" --model-path "${MODEL}" --output-dir "${CHECKPOINT}" --force --prune > "${ROOT}/logs/export_speed_${POLICY_ID}.log" 2>&1
fi
# All experiment GPUs are RTX 5090 (SM120).  Constraining JIT extensions to
# this architecture avoids rebuilding irrelevant SM80/86/90/100 variants.
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
if [ ! -s "${RUNS}/measured_0.json" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" --scenario prefill_decode --output-dir "${RUNS}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --warmup-iters 6 --iters 5 --single-phase main --single-output "${RUNS}/continuous_summary.json" --single-samples-dir "${RUNS}" --reuse-llm > "${ROOT}/logs/speed_${RUN_GROUP}_${POLICY_ID}_continuous.log" 2>&1
  exit 0
fi
for tag in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
  [ -s "${RUNS}/${tag}.json" ] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" --scenario prefill_decode --output-dir "${RUNS}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --warmup-iters 0 --iters 1 --single-phase main --single-output "${RUNS}/${tag}.json" > "${ROOT}/logs/speed_${RUN_GROUP}_${POLICY_ID}_${tag}.log" 2>&1
done
