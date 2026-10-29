#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
LOG_DIR="$ARTIFACT_DIR/logs"
RESULT_DIR="$ARTIFACT_DIR/results"
MODEL_DIR="$ARTIFACT_DIR/models/llama-2-7b-phase-switch-dense-nvfp4-marlin"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
ORIGINAL_MODEL="${ORIGINAL_MODEL:-/home/agent/wja/data/models/LLM-Research/llama-2-7b}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO_ROOT/vllm:$ARTIFACT_DIR:$CUTLASS_WRAPPER${PYTHONPATH:+:$PYTHONPATH}"
export PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

"$PYTHON_BIN" "$ARTIFACT_DIR/export_phase_switch_model.py" \
    --model-path "$ORIGINAL_MODEL" \
    --output-dir "$MODEL_DIR" \
    --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
    --force \
    > "$LOG_DIR/export_phase_switch.log" 2>&1

"$PYTHON_BIN" "$ARTIFACT_DIR/run_phase_switch_generate.py" \
    --model "$MODEL_DIR" \
    --output-json "$RESULT_DIR/generate_phase_switch.json" \
    --trace-json "$RESULT_DIR/switch_trace.json" \
    --repo-root "$REPO_ROOT" \
    --artifact-dir "$ARTIFACT_DIR" \
    --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
    > "$LOG_DIR/generate_phase_switch.log" 2>&1

echo "wrote $MODEL_DIR"
echo "wrote $RESULT_DIR/generate_phase_switch.json"
echo "wrote $RESULT_DIR/switch_trace.json"
