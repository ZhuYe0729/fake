#!/usr/bin/env bash
# Run p00/p01/p02/p71 on four GPUs after serial extension prewarm.
set -euo pipefail
: "${COSPAQ_GPUS:?source config.env first}"
IFS=',' read -r -a GPUS <<< "$COSPAQ_GPUS"
POLICIES=(p00 p01 p02 p71)
if (( ${#GPUS[@]} < ${#POLICIES[@]} )); then
  echo "four distinct GPUs are required for the smoke matrix" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$COSPAQ_RUN_ROOT/validation/logs"
mkdir -p "$LOG_ROOT"
pids=()
for index in "${!POLICIES[@]}"; do
  policy="${POLICIES[$index]}"
  gpu="${GPUS[$index]}"
  bash "$SCRIPT_DIR/run_smoke_policy.sh" "$policy" "$gpu" >"$LOG_ROOT/smoke_${policy}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

