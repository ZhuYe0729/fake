#!/usr/bin/env bash
# Check whether a policy OOM is removed by reserving less KV cache.
set -euo pipefail
POINT="${1:?point}"; GPU="${2:?gpu}"; UTIL="${3:-0.85}"
REPO="/root/wja/project/my/cospaq/fake"
ROOT="${REPO}/artifacts/debug/035_llama2_prefill_decode_e2e_speed_model"
SOURCE="${REPO}/artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver"
PYTHON="/home/agent/wja/miniconda3/envs/vllm/bin/python"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
CHECKPOINT="${ROOT}/checkpoints/point_$(printf '%03d' "${POINT}")"
[ -f "${CHECKPOINT}/model.safetensors" ] || CHECKPOINT="${SOURCE}/validation/prefill_decode/checkpoints/point_$(printf '%03d' "${POINT}")"
OUT="${ROOT}/headroom_probe/point_${POINT}_util_${UTIL}.json"
mkdir -p "$(dirname "${OUT}")"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" \
  --scenario prefill_decode --output-dir "$(dirname "${OUT}")" --gpu-memory-utilization "${UTIL}" \
  --warmup-iters 0 --iters 1 --single-phase main --single-output "${OUT}"
