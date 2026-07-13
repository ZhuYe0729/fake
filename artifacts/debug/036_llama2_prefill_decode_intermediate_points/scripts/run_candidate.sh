#!/usr/bin/env bash
# Materialize one dense-grid policy and measure it with the formal .85 protocol.
set -euo pipefail

POINT="${1:?dense-grid point index required}"
GPU="${2:?physical GPU index required}"
REPEATS="${REPEATS:-10}"
REPO="/root/wja/project/my/cospaq/fake"
ROOT="${REPO}/artifacts/debug/036_llama2_prefill_decode_intermediate_points"
POLICY="${ROOT}/prefill_decode/pareto/policies/point_$(printf '%03d' "${POINT}").json"
CHECKPOINT="${ROOT}/checkpoints/point_$(printf '%03d' "${POINT}")"
RUNS="${ROOT}/formal_util085/point_${POINT}/runs"
PYTHON="/home/agent/wja/miniconda3/envs/vllm/bin/python"
EXPORTER="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"

export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
mkdir -p "${ROOT}/logs" "${RUNS}"

if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${EXPORTER}" --policy-json "${POLICY}" \
    --output-dir "${CHECKPOINT}" --force --prune > "${ROOT}/logs/export_point_${POINT}.log" 2>&1
fi
for output in 1 80; do
  for kind in warmup $(seq 0 "$((REPEATS - 1))" | sed 's/^/measured_/'); do
    target="${RUNS}/${kind}_o${output}.json"
    [ -f "${target}" ] && continue
    phase="main"; [ "${output}" = 1 ] && phase="ttft"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" \
      --scenario prefill_decode --output-dir "${RUNS}" --gpu-memory-utilization 0.85 \
      --warmup-iters 0 --iters 1 --single-phase "${phase}" --single-output "${target}" \
      >> "${ROOT}/logs/speed_point_${POINT}.log" 2>&1
  done
done
