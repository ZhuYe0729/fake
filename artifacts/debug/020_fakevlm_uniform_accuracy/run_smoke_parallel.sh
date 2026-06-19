#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-5}" \
CALIB_SAMPLES="${CALIB_SAMPLES:-4}" \
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}" \
OVERWRITE="${OVERWRITE:-1}" \
bash "${ROOT}/run_accuracy_parallel.sh"
