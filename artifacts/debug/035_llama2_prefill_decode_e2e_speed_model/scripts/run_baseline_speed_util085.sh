#!/usr/bin/env bash
# Same workload and KV-cache reservation as the `.85` ours curve.
set -euo pipefail
REPO="/root/wja/project/my/cospaq/fake"
PYTHON="${PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
GPU="${GPU:-7}"
OUT="${REPO}/artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/baseline_speed_util085"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  "${REPO}/artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/benchmark_vllm_scenarios.py" \
  --scenarios prefill_decode --output-dir "${OUT}" --warmup-iters 1 --iters 10 --gpu-memory-utilization 0.85
