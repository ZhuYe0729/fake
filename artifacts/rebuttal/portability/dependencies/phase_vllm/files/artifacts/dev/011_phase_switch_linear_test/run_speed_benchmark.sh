#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
LOG_DIR="$ARTIFACT_DIR/logs_benchmark"
RESULT_DIR="$ARTIFACT_DIR/results_benchmark"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
DENSE_NVFP4_MODEL="${DENSE_NVFP4_MODEL:-$REPO_ROOT/artifacts/dev/003_vllm_integration_test/models/llama-2-7b-nvfp4-mytest}"
W4A16_MODEL="${W4A16_MODEL:-$REPO_ROOT/artifacts/dev/005_rtx5090_compression_sweep/models/llama-2-7b-w4a16-ours}"
PHASE_MODEL="${PHASE_MODEL:-$ARTIFACT_DIR/models/llama-2-7b-phase-switch-dense-nvfp4-marlin}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
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
    local scenario="$3"
    local input_len="$4"
    local output_len="$5"
    local batch_size="$6"
    local phase_flag="$7"
    local stem="${label}_${scenario}_i${input_len}_o${output_len}_bs${batch_size}"
    local trace_args=()
    if [[ "$phase_flag" == "1" ]]; then
        trace_args=(--phase-switch --trace-json "$RESULT_DIR/${stem}_trace.json")
    fi
    "$PYTHON_BIN" "$ARTIFACT_DIR/benchmark_generate.py" \
        --model "$model" \
        --label "$label" \
        --scenario "$scenario" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --batch-size "$batch_size" \
        --output-json "$RESULT_DIR/${stem}.json" \
        --repo-root "$REPO_ROOT" \
        --artifact-dir "$ARTIFACT_DIR" \
        --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
        "${trace_args[@]}" \
        > "$LOG_DIR/${stem}.log" 2>&1
}

run_scenario() {
    local scenario="$1"
    local input_len="$2"
    local output_len="$3"
    local batch_size="$4"
    run_one dense_nvfp4 "$DENSE_NVFP4_MODEL" "$scenario" "$input_len" "$output_len" "$batch_size" 0
    run_one w4a16_ours "$W4A16_MODEL" "$scenario" "$input_len" "$output_len" "$batch_size" 0
    run_one phase_switch "$PHASE_MODEL" "$scenario" "$input_len" "$output_len" "$batch_size" 1
}

run_scenario prefill_heavy 512 32 16
run_scenario decode_heavy 32 128 16
run_scenario balanced 256 256 16

"$PYTHON_BIN" "$ARTIFACT_DIR/summarize_speed_benchmark.py" \
    --result-dir "$RESULT_DIR" \
    --output-json "$RESULT_DIR/summary.json" \
    --output-md "$RESULT_DIR/summary.md"

echo "wrote $RESULT_DIR/summary.json"
echo "wrote $RESULT_DIR/summary.md"
