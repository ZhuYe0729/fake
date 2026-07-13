#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ROOT="/home/agent/wja/project/my/cospaq/test/vllm"
CUTLASS_ROOT="/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-7}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.8}"
BENCH_ONE="${VLLM_ROOT}/artifacts/dev/013_phase_hetero_speed_matrix/benchmark_one.py"
PREFILL_BENCH="/root/wja/project/my/cospaq/fake/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py"
for scenario in prefill_only prefill_decode; do
  mapfile -t points < <("${PYTHON}" -c "import json;[print(x['point_index']) for x in json.load(open('${ROOT}/validation/${scenario}/selection.json'))]")
  for point in "${points[@]}"; do
    checkpoint="${ROOT}/validation/${scenario}/checkpoints/point_$(printf '%03d' "${point}")"
    [ -f "${checkpoint}/model.safetensors" ] || { echo "missing ${checkpoint}" >&2; exit 1; }
    speed_dir="speed"
    [ "${scenario}" = prefill_decode ] && speed_dir="speed_mem08"
    out="${ROOT}/validation/${scenario}/${speed_dir}/point_${point}"; mkdir -p "${out}/runs"
    if [ "${scenario}" = prefill_only ]; then
      for i in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
        [ -f "${out}/runs/${i}.json" ] && continue
        CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${PREFILL_BENCH}" --checkpoint "${checkpoint}" --output-json "${out}/runs/${i}.json"
      done
    else
      for output in 1 80; do
        for i in warmup measured_0 measured_1 measured_2 measured_3 measured_4 measured_5 measured_6 measured_7 measured_8 measured_9; do
          [ -f "${out}/runs/${i}_o${output}.json" ] && continue
          CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH_ONE}" --model "${checkpoint}" --label "pareto_point_${point}" --category phase_hetero --scenario prefill_decode --input-len 2048 --output-len "${output}" --batch-size 16 --output-json "${out}/runs/${i}_o${output}.json" --repo-root "${VLLM_ROOT}" --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" --cutlass-wrapper-path "${CUTLASS_ROOT}" --phase-hetero --max-num-batched-tokens 32768 --gpu-memory-utilization "${DECODE_GPU_MEMORY_UTILIZATION}"
        done
      done
    fi
  done
done
