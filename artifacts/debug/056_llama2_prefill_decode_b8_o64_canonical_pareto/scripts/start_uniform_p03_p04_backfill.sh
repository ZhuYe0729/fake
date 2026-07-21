#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/agent/wja/project/my/cospaq/fake/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto"
LAUNCHER="${ROOT}/scripts/launch_uniform_p03_p04_backfill.sh"
LOG_DIR="${ROOT}/logs/uniform_task_backfill"
mkdir -p "${LOG_DIR}"

nohup "${LAUNCHER}" 0 p03 cnn_dm_1000 0 360 p04 cnn_dm_1000 0 360 p03 dsum 0 360 p04 dsum 0 360 >"${LOG_DIR}/gpu0.log" 2>&1 &
nohup "${LAUNCHER}" 2 p03 cnn_dm_1000 360 720 p04 cnn_dm_1000 360 720 p03 dsum 360 720 p04 dsum 360 720 >"${LOG_DIR}/gpu2.log" 2>&1 &
nohup "${LAUNCHER}" 3 p03 cnn_dm_1000 720 1000 p04 cnn_dm_1000 720 1000 p03 dsum 720 1080 p04 dsum 720 1080 >"${LOG_DIR}/gpu3.log" 2>&1 &
nohup "${LAUNCHER}" 4 p03 dsum 1080 1440 p04 dsum 1080 1440 p03 IWSLT 0 100 >"${LOG_DIR}/gpu4.log" 2>&1 &
nohup "${LAUNCHER}" 5 p04 IWSLT 0 100 p03 dsum 1440 1500 p04 dsum 1440 1500 >"${LOG_DIR}/gpu5.log" 2>&1 &
nohup "${LAUNCHER}" 6 p03 IWSLT 100 200 p04 IWSLT 100 200 p03 IWSLT 200 300 >"${LOG_DIR}/gpu6.log" 2>&1 &
nohup "${LAUNCHER}" 7 p04 IWSLT 200 300 p03 IWSLT 300 333 p04 IWSLT 300 333 >"${LOG_DIR}/gpu7.log" 2>&1 &
