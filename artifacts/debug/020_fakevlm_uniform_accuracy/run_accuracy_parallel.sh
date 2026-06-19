#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS=(dense_bf16 sparse_bf16 dense_nvfp4 sparse_nvfp4 marlin_weight_only dense_nvfp4_prefill_marlin_decode)
GPUS=(7 6 5 4 3 2)

MODEL_PATH="${MODEL_PATH:-/home/agent/wja/data/models/lingcco/fakeVLM}"
TEST_JSON_FILE="${TEST_JSON_FILE:-/home/agent/wja/data/datasets/lingcco/FakeClue/data_json/test.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/agent/wja/data/datasets/lingcco/FakeClue/test/test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-1}"
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DECODE_M_THRESHOLD="${DECODE_M_THRESHOLD:-8}"
SEED="${SEED:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${ROOT}/summary"

for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  gpu="${GPUS[$idx]}"
  log="${ROOT}/logs/${method}.log"
  status="${ROOT}/status/${method}.json"
  (
    start_ts="$(date -Is)"
    printf '{"method":"%s","gpu":%s,"start_time":"%s","state":"running"}\n' "${method}" "${gpu}" "${start_ts}" > "${status}"
    args=(
      conda run -n cospaq python "${ROOT}/eval_fakevlm_uniform_accuracy.py"
      --method "${method}"
      --model-path "${MODEL_PATH}"
      --test-json-file "${TEST_JSON_FILE}"
      --image-root "${IMAGE_ROOT}"
      --output-root "${ROOT}"
      --batch-size "${BATCH_SIZE}"
      --workers "${WORKERS}"
      --calib-samples "${CALIB_SAMPLES}"
      --calib-batch-size "${CALIB_BATCH_SIZE}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --decode-m-threshold "${DECODE_M_THRESHOLD}"
      --seed "${SEED}"
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
      printf '{"method":"%s","gpu":%s,"start_time":"%s","end_time":"%s","state":"done","exit_code":0}\n' "${method}" "${gpu}" "${start_ts}" "${end_ts}" > "${status}"
    else
      error_tail="$(tail -40 "${log}" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
      printf '{"method":"%s","gpu":%s,"start_time":"%s","end_time":"%s","state":"failed","exit_code":%s,"error":%s}\n' "${method}" "${gpu}" "${start_ts}" "${end_ts}" "${exit_code}" "${error_tail}" > "${status}"
    fi
    exit "${exit_code}"
  ) > "${log}" 2>&1 &
done

failed=0
for job in $(jobs -p); do
  if ! wait "${job}"; then
    failed=1
  fi
done

conda run -n cospaq python "${ROOT}/summarize_accuracy.py" --output-root "${ROOT}"
exit "${failed}"
