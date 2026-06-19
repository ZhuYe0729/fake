#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${ROOT}/code"

MODEL_PATH="${MODEL_PATH:-/home/agent/wja/data/models/lingcco/fakeVLM}"
TEST_JSON_FILE="${TEST_JSON_FILE:-/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/agent/wja/data/datasets/lingcco/FakeClue/test/test}"
BATCH_SIZES_STR="${BATCH_SIZES:-1 2 4 8 16}"
GPUS_STR="${GPUS:-0 1}"
WORKERS="${WORKERS:-1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
MANUAL_WARMUP="${MANUAL_WARMUP:-3}"
MANUAL_ITERS="${MANUAL_ITERS:-10}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"
FAMILIES_STR="${FAMILIES:-manual_profile latency_model uniform_dense_bf16 uniform_dense_nvfp4 uniform_sparse_bf16 uniform_sparse_nvfp4}"

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${ROOT}/summary"

read -r -a BATCH_SIZES <<< "${BATCH_SIZES_STR}"
read -r -a GPUS <<< "${GPUS_STR}"
read -r -a FAMILIES <<< "${FAMILIES_STR}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs configured." >&2
  exit 1
fi

failed=0
active=0
job_index=0
for batch_size in "${BATCH_SIZES[@]}"; do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  log="${ROOT}/logs/batch_${batch_size}.log"
  status="${ROOT}/status/batch_${batch_size}.json"
  (
    start_ts="$(date -Is)"
    printf '{"batch_size":%s,"gpu":%s,"start_time":"%s","state":"running"}\n' "${batch_size}" "${gpu}" "${start_ts}" > "${status}"
    args=(
      conda run -n cospaq python "${CODE_ROOT}/run_fakevlm_prefill_speed.py"
      --model-path "${MODEL_PATH}"
      --test-json-file "${TEST_JSON_FILE}"
      --image-root "${IMAGE_ROOT}"
      --output-root "${ROOT}"
      --batch-size "${batch_size}"
      --workers "${WORKERS}"
      --warmup "${WARMUP}"
      --iters "${ITERS}"
      --manual-warmup "${MANUAL_WARMUP}"
      --manual-iters "${MANUAL_ITERS}"
      --seed "${SEED}"
      --families "${FAMILIES[@]}"
    )
    if [[ -n "${SAMPLE_LIMIT}" ]]; then
      args+=(--sample-limit "${SAMPLE_LIMIT}")
    fi
    if [[ "${OVERWRITE}" == "1" ]]; then
      args+=(--overwrite)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${args[@]}"
    exit_code=$?
    end_ts="$(date -Is)"
    if [[ ${exit_code} -eq 0 ]]; then
      printf '{"batch_size":%s,"gpu":%s,"start_time":"%s","end_time":"%s","state":"done","exit_code":0}\n' "${batch_size}" "${gpu}" "${start_ts}" "${end_ts}" > "${status}"
    else
      error_tail="$(tail -40 "${log}" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
      printf '{"batch_size":%s,"gpu":%s,"start_time":"%s","end_time":"%s","state":"failed","exit_code":%s,"error":%s}\n' "${batch_size}" "${gpu}" "${start_ts}" "${end_ts}" "${exit_code}" "${error_tail}" > "${status}"
    fi
    exit "${exit_code}"
  ) > "${log}" 2>&1 &

  active=$((active + 1))
  job_index=$((job_index + 1))
  if [[ ${active} -ge ${#GPUS[@]} ]]; then
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  fi
done

while [[ ${active} -gt 0 ]]; do
  if ! wait -n; then
    failed=1
  fi
  active=$((active - 1))
done

conda run -n cospaq python "${CODE_ROOT}/summarize_prefill_speed.py" --output-root "${ROOT}"
exit "${failed}"
