#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-$ARTIFACT_DIR/results_quality_phase}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_DIR/logs_quality_phase}"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
BF16_MODEL="${BF16_MODEL:-/home/agent/wja/data/models/LLM-Research/llama-2-7b}"
DENSE_NVFP4_MODEL="${DENSE_NVFP4_MODEL:-$REPO_ROOT/artifacts/dev/003_vllm_integration_test/models/llama-2-7b-nvfp4-mytest}"
W4A16_MODEL="${W4A16_MODEL:-$REPO_ROOT/artifacts/dev/005_rtx5090_compression_sweep/models/llama-2-7b-w4a16-ours}"
PHASE_MODEL="${PHASE_MODEL:-$ARTIFACT_DIR/models/llama-2-7b-phase-switch-dense-nvfp4-marlin}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO_ROOT/vllm:$ARTIFACT_DIR:$CUTLASS_WRAPPER${PYTHONPATH:+:$PYTHONPATH}"
export PHASE_SWITCH_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH="$CUTLASS_WRAPPER"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

run_one() {
    local label="$1"
    local model="$2"
    local phase_flag="$3"
    local args=()
    if [[ "$phase_flag" == "1" ]]; then
        args=(--phase-switch)
    fi
    "$PYTHON_BIN" "$ARTIFACT_DIR/eval_quality_models.py" \
        --model "$model" \
        --label "$label" \
        --output-json "$RESULT_DIR/${label}_quality.json" \
        --repo-root "$REPO_ROOT" \
        --artifact-dir "$ARTIFACT_DIR" \
        --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
        "${args[@]}" \
        > "$LOG_DIR/${label}.log" 2>&1
}

run_one bf16 "$BF16_MODEL" 0
run_one dense_nvfp4 "$DENSE_NVFP4_MODEL" 0
run_one w4a16_ours "$W4A16_MODEL" 0
run_one phase_switch_overlap "$PHASE_MODEL" 1

"$PYTHON_BIN" "$ARTIFACT_DIR/summarize_quality_models.py" \
    --result-dir "$RESULT_DIR" \
    --output-json "$RESULT_DIR/summary.json" \
    --output-md "$RESULT_DIR/summary.md"

echo "wrote $RESULT_DIR/summary.json"
echo "wrote $RESULT_DIR/summary.md"
