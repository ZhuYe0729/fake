#!/usr/bin/env bash
# Measure the two uniform references absent from the selected candidate set.
# Run only after run_selected_vllm_speed.sh has released the assigned GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/root/wja/project/my/cospaq/fake"
VLLM_ROOT="/home/agent/wja/project/my/cospaq/test/vllm"
CUTLASS_ROOT="${REPO}/fake/kernels/cutlass/cutlass_wrapper"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORT_PYTHON="${EXPORT_PYTHON:-/home/agent/wja/miniconda3/envs/cospaq/bin/python}"
GPU="${GPU:-7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
EXPORTER="${REPO}/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
BENCH_ONE="${VLLM_ROOT}/artifacts/dev/013_phase_hetero_speed_matrix/benchmark_one.py"
POLICY_ROOT="${REPO}/artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/policies/prefill_decode"

# dense_bf16 and dense_nvfp4 are already point_0 and point_11, respectively.
# sparse-nvfp4 has no legal decode kernel, so p03 uses its supported phase pair.
for spec in "sparse_bf16 p02" "sparse_nvfp4_prefill_dense_nvfp4_decode p03" "w4a16_ours p04"; do
  read -r label policy_id <<<"${spec}"
  out="${ROOT}/validation/prefill_decode/uniform_references/${label}"
  checkpoint="${out}/checkpoint"
  mkdir -p "${out}/runs"
  if [ ! -f "${checkpoint}/model.safetensors" ]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${EXPORT_PYTHON}" "${EXPORTER}" \
      --policy-json "${POLICY_ROOT}/${policy_id}.json" --output-dir "${checkpoint}" --force --prune \
      >"${out}/export.log" 2>&1
  fi
  for output in 1 80; do
    for i in warmup measured_0 measured_1 measured_2 measured_3 measured_4 measured_5 measured_6 measured_7 measured_8 measured_9; do
      [ -f "${out}/runs/${i}_o${output}.json" ] && continue
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${BENCH_ONE}" \
        --model "${checkpoint}" --label "uniform_${label}" --category uniform_reference \
        --scenario prefill_decode --input-len 2048 --output-len "${output}" --batch-size 16 \
        --output-json "${out}/runs/${i}_o${output}.json" --repo-root "${VLLM_ROOT}" \
        --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" \
        --cutlass-wrapper-path "${CUTLASS_ROOT}" --phase-hetero --max-num-batched-tokens 32768 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    done
  done
done
