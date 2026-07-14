#!/usr/bin/env bash
# Run one persistent-vLLM PMPD shard for a closed Pareto checkpoint.
set -euo pipefail
if [ "$#" -ne 5 ]; then echo "usage: $0 POINT_ID DATASET BEGIN END PHYSICAL_GPU" >&2; exit 2; fi
POINT_ID="$1"; DATASET="$2"; BEGIN="$3"; END="$4"; GPU="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="/home/agent/wja/project/my/cospaq/fake"
VLLM_ROOT="/home/agent/wja/project/my/cospaq/test/vllm"
CUTLASS_ROOT="${REPO}/fake/kernels/cutlass/cutlass_wrapper"
PYTHON_BIN="/home/agent/wja/miniconda3/envs/vllm/bin/python"
CHECKPOINT="${ROOT}/closure/checkpoints/${POINT_ID}"
OUT_DIR="${ROOT}/closure/tasks/${POINT_ID}/shards/${DATASET}/shard_${BEGIN}_${END}"
LABEL="pareto_${POINT_ID}"
mkdir -p "${OUT_DIR}"
export PYTHONPATH="${VLLM_ROOT}/vllm:${VLLM_ROOT}:${CUTLASS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST=12.0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0 PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
export PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}" SPARSE_BF16_MYTEST_CUTLASS_WRAPPER_PATH="${CUTLASS_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py" \
  --dataset "${DATASET}" --question-begin "${BEGIN}" --question-end "${END}" --batch-size "${BATCH_SIZE:-16}" \
  --model-path "${CHECKPOINT}" --model-id "${LABEL}" --output-dir "${OUT_DIR}" \
  --repo-root "${VLLM_ROOT}" --artifact-dir "${VLLM_ROOT}/artifacts/dev/011_phase_switch_linear_test" \
  --phase-artifact-dir "${VLLM_ROOT}/artifacts/dev/012_phase_hetero_linear" --cutlass-wrapper-path "${CUTLASS_ROOT}" \
  --phase-hetero --max-num-batched-tokens 32768 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --iwslt-filter-tokenizer /home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf --skip-metrics
