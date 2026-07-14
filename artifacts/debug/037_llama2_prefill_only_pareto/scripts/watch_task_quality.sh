#!/usr/bin/env bash
# Keep the recoverable task-quality launcher alive until all expected shards exist.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/agent/wja/miniconda3/envs/cospaq/bin/python"
LOG_DIR="${ROOT}/task_quality/logs"
mkdir -p "${LOG_DIR}"

complete_count() {
  find "${ROOT}/task_quality/shards" -type f -name 'ours_point_*_prefill_only-fp16.jsonl' 2>/dev/null | \
    "${PYTHON}" -c 'import sys; print(sum(1 for _ in sys.stdin))'
}

while true; do
  count="$(complete_count)"
  if [ "${count}" -ge 36 ]; then
    echo "$(date -u +%FT%TZ) complete ${count}/36" >> "${LOG_DIR}/watchdog.log"
    exit 0
  fi
  if ! pgrep -f '[r]un_task_quality.py --gpus 1,2,4 --points 6,9,15' >/dev/null; then
    echo "$(date -u +%FT%TZ) restart ${count}/36" >> "${LOG_DIR}/watchdog.log"
    nohup "${PYTHON}" "${ROOT}/scripts/run_task_quality.py" --gpus 1,2,4 --points 6,9,15 --batch-size 4 \
      >> "${LOG_DIR}/watchdog_launcher.log" 2>&1 < /dev/null &
  fi
  sleep 60
done
