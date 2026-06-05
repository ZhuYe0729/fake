#!/usr/bin/env bash
set -euo pipefail

MODELS=("Qwen3.5-4B" "Qwen3.5-9B" "Qwen3.5-27B")
GPU_LIST="${GPU_LIST:-1 2 3}"
read -r -a GPUS <<< "${GPU_LIST}"

if [ "${#GPUS[@]}" -lt "${#MODELS[@]}" ]; then
  echo "Need at least ${#MODELS[@]} GPUs in GPU_LIST, got: ${GPU_LIST}" >&2
  exit 1
fi

mkdir -p artifacts/results/benchmarks/module/logs
mkdir -p /tmp/qwen35_mplconfig

pids=()
for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPUS[$idx]}"
  slug="$(echo "${model}" | tr '[:upper:]' '[:lower:]' | tr -d '.' | tr '-' '_')"
  out_dir="artifacts/results/benchmarks/module/${model}/kernel"
  csv_path="${out_dir}/${slug}_module_kernel_curves.csv"
  log_path="artifacts/results/benchmarks/module/logs/${slug}_module_kernel_bench.log"

  mkdir -p "${out_dir}"
  echo "Launching ${model} on physical GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/bench_qwen3_5_module_kernels.py \
    --model-name "${model}" \
    --gpu 0 \
    --output "${csv_path}" \
    --warmup 5 \
    --iters 20 > "${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

for model in "${MODELS[@]}"; do
  slug="$(echo "${model}" | tr '[:upper:]' '[:lower:]' | tr -d '.' | tr '-' '_')"
  out_dir="artifacts/results/benchmarks/module/${model}/kernel"
  csv_path="${out_dir}/${slug}_module_kernel_curves.csv"
  png_path="${out_dir}/${slug}_module_kernel_latency_curves.png"

  MPLCONFIGDIR=/tmp/qwen35_mplconfig python scripts/visualize_qwen3_5_module_kernel_curves.py \
    --input "${csv_path}" \
    --output "${png_path}"
done
