#!/usr/bin/env bash
# Export one frozen 034 policy, then collect one warmup and five fresh E2E runs.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 POINT GPU" >&2
  exit 2
fi

POINT="$1"
GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${ROOT%/037_llama2_prefill_only_pareto}/034_llama2_7b_chat_wikitext_pareto_solver"
REPO="/root/wja/project/my/cospaq/fake"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
BENCH_PY="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
EXPORT_PY="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
POLICY="${SOURCE}/prefill_only/pareto/policies/point_$(printf '%03d' "${POINT}").json"
CHECKPOINT="${ROOT}/checkpoints/point_$(printf '%03d' "${POINT}")"
RUNS="${ROOT}/measurements/point_${POINT}/runs"

mkdir -p "${RUNS}" "${ROOT}/logs"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORT_PY}" \
    --policy-json "${POLICY}" --output-dir "${CHECKPOINT}" --force --prune \
    > "${ROOT}/logs/export_point_${POINT}.log" 2>&1
fi
for tag in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
  [ -f "${RUNS}/${tag}.json" ] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH_PY}" \
    --checkpoint "${CHECKPOINT}" --batch 8 --input-seq 2048 --output-seq 1 \
    --gpu-memory-utilization 0.9 --output-json "${RUNS}/${tag}.json" \
    > "${ROOT}/logs/speed_point_${POINT}_${tag}.log" 2>&1
done
