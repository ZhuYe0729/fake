#!/usr/bin/env bash
# Historical formal decode protocol: scenario max_model_len, phase runtime flags,
# GPU memory utilization 0.9, and a fresh process for every timing sample.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/root/wja/project/my/cospaq/fake"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-7}"
POINTS="${POINTS:-0,3,6,11}"
REPEATS="${REPEATS:-10}"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"

export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1

IFS=',' read -ra point_list <<<"${POINTS}"
for point in "${point_list[@]}"; do
  checkpoint="${ROOT}/validation/prefill_decode/checkpoints/point_$(printf '%03d' "${point}")"
  [ -f "${checkpoint}/model.safetensors" ] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
  runs="${ROOT}/validation/prefill_decode/speed_official/point_${point}/runs"
  mkdir -p "${runs}"
  for spec in "ttft 1" "main 80"; do
    read -r phase output <<<"${spec}"
    for kind in warmup $(seq 0 "$((REPEATS - 1))" | sed 's/^/measured_/'); do
      target="${runs}/${kind}_o${output}.json"
      [ -f "${target}" ] && continue
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH}" \
        --checkpoint "${checkpoint}" --scenario prefill_decode --output-dir "${runs}" \
        --gpu-memory-utilization 0.9 --warmup-iters 0 --iters 1 \
        --single-phase "${phase}" --single-output "${target}"
    done
  done
done
