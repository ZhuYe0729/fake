#!/usr/bin/env bash
# Fresh-process E2E closure for one solved canonical Pareto policy.
set -euo pipefail

if [ "$#" -ne 2 ]; then echo "usage: $0 POLICY_ID PHYSICAL_GPU" >&2; exit 2; fi
POLICY_ID="$1"
GPU="$2"
REPEATS="${REPEATS:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
KV_CACHE_ARGS=()
if [ -n "${KV_CACHE_MEMORY_BYTES}" ]; then
  KV_CACHE_ARGS=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi
RUN_GROUP="${RUN_GROUP:-runs}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/root/wja/project/my/cospaq/fake"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORTER="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
POLICY_DIR="${POLICY_DIR:-${ROOT}/llama2_7b_chat/pareto/policies}"
POLICY="${POLICY_DIR}/${POLICY_ID}.json"
OUT="${ROOT}/llama2_7b_chat/validation/speed/${POLICY_ID}"
CHECKPOINT="${OUT}/checkpoint"
RUNS="${OUT}/${RUN_GROUP}"

mkdir -p "${RUNS}" "${ROOT}/logs/pareto_speed"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${EXPORTER}" --policy-json "${POLICY}" \
    --output-dir "${CHECKPOINT}" --force > "${ROOT}/logs/pareto_speed/export_${POLICY_ID}.log" 2>&1
fi

export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
for spec in "ttft 1" "main 80"; do
  read -r phase output <<<"${spec}"
  for kind in warmup $(seq 0 "$((REPEATS - 1))" | sed 's/^/measured_/'); do
    target="${RUNS}/${kind}_o${output}.json"
    [ -f "${target}" ] && continue
    # Deliberately no --reuse-llm: this is the formal 035 fresh-process protocol.
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" \
      --scenario prefill_decode --output-dir "${RUNS}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --warmup-iters 0 --iters 1 --single-phase "${phase}" --single-output "${target}" \
      "${KV_CACHE_ARGS[@]}" \
      >> "${ROOT}/logs/pareto_speed/measure_${POLICY_ID}.log" 2>&1
  done
done
