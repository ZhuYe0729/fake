#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-$ARTIFACT_DIR/results_phase_overhead_debug}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_DIR/logs_phase_overhead_debug}"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
PHASE_MODEL="${PHASE_MODEL:-$ARTIFACT_DIR/models/llama-2-7b-phase-switch-dense-nvfp4-marlin}"

SCENARIO="${SCENARIO:-phase_overhead_debug_i2048_o80_bs16}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

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
    local out_len="$2"
    local trace="$3"
    local gc_every="$4"
    local wait_once="$5"
    local gc_decode="$6"
    local stem="${label}_${SCENARIO}_o${out_len}"

    PHASE_SWITCH_TRACE="$trace" \
    PHASE_SWITCH_GC_EVERY_APPLY="$gc_every" \
    PHASE_SWITCH_WAIT_ONCE="$wait_once" \
    PHASE_SWITCH_GC_DECODE="$gc_decode" \
    "$PYTHON_BIN" "$ARTIFACT_DIR/benchmark_generate.py" \
        --model "$PHASE_MODEL" \
        --label "$label" \
        --scenario "$SCENARIO" \
        --input-len "$INPUT_LEN" \
        --output-len "$out_len" \
        --batch-size "$BATCH_SIZE" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --output-json "$RESULT_DIR/${stem}.json" \
        --trace-json "$RESULT_DIR/${stem}_trace.json" \
        --repo-root "$REPO_ROOT" \
        --artifact-dir "$ARTIFACT_DIR" \
        --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
        --phase-switch \
        > "$LOG_DIR/${stem}.log" 2>&1
}

run_pair() {
    local label="$1"
    local trace="$2"
    local gc_every="$3"
    local wait_once="$4"
    local gc_decode="$5"
    run_one "$label" 1 "$trace" "$gc_every" "$wait_once" "$gc_decode"
    run_one "$label" "$OUTPUT_LEN" "$trace" "$gc_every" "$wait_once" "$gc_decode"
}

run_pair phase_baseline 1 1 0 1
run_pair phase_no_trace 0 1 0 1
run_pair phase_no_decode_gc 1 1 0 0
run_pair phase_wait_once 1 1 1 1
run_pair phase_combined 0 1 1 0

"$PYTHON_BIN" "$ARTIFACT_DIR/summarize_phase_overhead_debug.py" \
    --result-dir "$RESULT_DIR" \
    --scenario "$SCENARIO" \
    --output-len "$OUTPUT_LEN" \
    --output-json "$RESULT_DIR/summary.json" \
    --output-md "$RESULT_DIR/summary.md"

echo "wrote $RESULT_DIR/summary.json"
echo "wrote $RESULT_DIR/summary.md"
