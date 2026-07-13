#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/benchmark_vllm_scenarios.py" \
  --methods dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4 \
  --scenarios prefill_only,prefill_decode \
  "$@"
