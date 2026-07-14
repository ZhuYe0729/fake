#!/usr/bin/env bash
# Export one frozen policy and collect five fresh vLLM prefill-only samples.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 POLICY_ID PHYSICAL_GPU" >&2; exit 2; fi
POLICY_ID="$1"; GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/home/agent/wja/project/my/cospaq/fake"
MODEL="/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORT="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
POLICY="${ROOT}/policies/prefill_only/${POLICY_ID}.json"
CHECKPOINT="${ROOT}/speed_calibration/checkpoints/${POLICY_ID}"
RUNS="${ROOT}/speed_calibration/runs/${POLICY_ID}"
mkdir -p "${RUNS}" "${ROOT}/logs"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORT}" --policy-json "${POLICY}" --model-path "${MODEL}" --output-dir "${CHECKPOINT}" --force --prune > "${ROOT}/logs/export_speed_${POLICY_ID}.log" 2>&1
fi
for tag in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
  [ -s "${RUNS}/${tag}.json" ] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" --batch 8 --input-seq 2048 --output-seq 1 --gpu-memory-utilization 0.9 --output-json "${RUNS}/${tag}.json" > "${ROOT}/logs/speed_${POLICY_ID}_${tag}.log" 2>&1
done
