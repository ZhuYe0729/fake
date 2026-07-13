#!/usr/bin/env bash
# One isolated formal-protocol E2E calibration point. Intended for one GPU.
set -euo pipefail

POINT="${1:?point index required}"
GPU="${2:?GPU index required}"
REPEATS="${REPEATS:-3}"
RUN_GROUP="${RUN_GROUP:-measurements}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
REPO="/root/wja/project/my/cospaq/fake"
SOURCE="${REPO}/artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver"
ROOT="${REPO}/artifacts/debug/035_llama2_prefill_decode_e2e_speed_model"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORTER="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
POLICY="${SOURCE}/prefill_decode/pareto/policies/point_$(printf '%03d' "${POINT}").json"
CHECKPOINT="${ROOT}/checkpoints/point_$(printf '%03d' "${POINT}")"
FALLBACK_CHECKPOINT="${SOURCE}/validation/prefill_decode/checkpoints/point_$(printf '%03d' "${POINT}")"
RUNS="${ROOT}/${RUN_GROUP}/point_${POINT}/runs"

export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1

mkdir -p "${ROOT}/logs" "${RUNS}"
if [ ! -f "${CHECKPOINT}/model.safetensors" ] && [ -f "${FALLBACK_CHECKPOINT}/model.safetensors" ]; then
  CHECKPOINT="${FALLBACK_CHECKPOINT}"
fi
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${EXPORTER}" --policy-json "${POLICY}" \
    --output-dir "${CHECKPOINT}" --force --prune > "${ROOT}/logs/export_point_${POINT}.log" 2>&1
fi
for spec in "ttft 1" "main 80"; do
  read -r phase output <<<"${spec}"
  for kind in warmup $(seq 0 "$((REPEATS - 1))" | sed 's/^/measured_/'); do
    target="${RUNS}/${kind}_o${output}.json"
    [ -f "${target}" ] && continue
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" \
      --scenario prefill_decode --output-dir "${RUNS}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --warmup-iters 0 --iters 1 --single-phase "${phase}" --single-output "${target}" \
      >> "${ROOT}/logs/speed_point_${POINT}.log" 2>&1
  done
done
