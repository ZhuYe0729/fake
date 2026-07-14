#!/usr/bin/env bash
# Fresh E2E speed plus fixed WikiText NLL for one predicted Pareto point.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 POINT_INDEX PHYSICAL_GPU" >&2; exit 2; fi
POINT="$1"; GPU="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="/home/agent/wja/project/my/cospaq/fake"
MODEL="/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"; VLLM_PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORT="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"; BENCH="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
POLICY="${ROOT}/pareto/policies/point_$(printf '%03d' "${POINT}").json"; CHECKPOINT="${ROOT}/closure/checkpoints/point_${POINT}"; RUNS="${ROOT}/closure/speed/point_${POINT}/runs"; NLL="${ROOT}/closure/nll/point_${POINT}.csv"
mkdir -p "${RUNS}" "${ROOT}/closure/nll" "${ROOT}/logs"
if [ ! -f "${CHECKPOINT}/model.safetensors" ]; then CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORT}" --policy-json "${POLICY}" --model-path "${MODEL}" --output-dir "${CHECKPOINT}" --force --prune > "${ROOT}/logs/closure_export_${POINT}.log" 2>&1; fi
for tag in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
  [ -s "${RUNS}/${tag}.json" ] && continue
  CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_PYTHON}" "${BENCH}" --checkpoint "${CHECKPOINT}" --batch 8 --input-seq 2048 --output-seq 1 --gpu-memory-utilization 0.9 --output-json "${RUNS}/${tag}.json" > "${ROOT}/logs/closure_speed_${POINT}_${tag}.log" 2>&1
done
rm -rf "${CHECKPOINT}"
if [ ! -s "${NLL}" ]; then CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${ROOT}/scripts/evaluate_wikitext_nll.py" --policy "point_${POINT}" --policy-json "${POLICY}" --gpu 0 --blocks 100 --batch-size 1 --dense-reference-json "${ROOT}/nll_shards/p00.csv" --output-csv "${NLL}" > "${ROOT}/logs/closure_nll_${POINT}.log" 2>&1; fi
