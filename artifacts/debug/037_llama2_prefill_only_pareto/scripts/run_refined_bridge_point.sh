#!/usr/bin/env bash
# Export and measure a refined policy under the frozen prefill-only protocol.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 LABEL POLICY_JSON GPU" >&2
  exit 2
fi

LABEL=$1
POLICY=$2
GPU=$3
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=/home/agent/wja/project/my/cospaq/fake
EXPORT_PYTHON=/home/agent/wja/miniconda3/envs/cospaq/bin/python
VLLM_PYTHON=/home/agent/wja/miniconda3/envs/vllm/bin/python
EXPORT_PY=$REPO/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py
BENCH_PY=$REPO/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_baseline_one.py
CHECKPOINT=$ROOT/refined_bridge/checkpoints/$LABEL
RUNS=$ROOT/refined_bridge/measurements/$LABEL/runs
LOGS=$ROOT/refined_bridge/logs

mkdir -p "$RUNS" "$LOGS"
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$EXPORT_PYTHON" "$EXPORT_PY" \
    --policy-json "$POLICY" --output-dir "$CHECKPOINT" --force --prune \
    > "$LOGS/export_${LABEL}.log" 2>&1
fi
for tag in warmup measured_0 measured_1 measured_2 measured_3 measured_4; do
  [ -f "$RUNS/$tag.json" ] && continue
  CUDA_VISIBLE_DEVICES=$GPU "$VLLM_PYTHON" "$BENCH_PY" \
    --checkpoint "$CHECKPOINT" --batch 8 --input-seq 2048 --output-seq 1 \
    --gpu-memory-utilization 0.9 --output-json "$RUNS/$tag.json" \
    > "$LOGS/speed_${LABEL}_${tag}.log" 2>&1
done
