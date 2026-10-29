#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-$ARTIFACT_DIR/logs_decomposed}"
RESULT_DIR="${RESULT_DIR:-$ARTIFACT_DIR/results_decomposed}"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
DENSE_NVFP4_MODEL="${DENSE_NVFP4_MODEL:-$REPO_ROOT/artifacts/dev/003_vllm_integration_test/models/llama-2-7b-nvfp4-mytest}"
W4A16_MODEL="${W4A16_MODEL:-$REPO_ROOT/artifacts/dev/005_rtx5090_compression_sweep/models/llama-2-7b-w4a16-ours}"
PHASE_MODEL="${PHASE_MODEL:-$ARTIFACT_DIR/models/llama-2-7b-phase-switch-dense-nvfp4-marlin}"

SCENARIO="${SCENARIO:-prefill_heavy_i512_o32_bs16}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

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
    local out_len="$3"
    local phase_flag="$4"
    local sync_copy="$5"
    local stem="${label}_${SCENARIO}_o${out_len}"
    local trace_args=()
    if [[ "$phase_flag" == "1" ]]; then
        trace_args=(--phase-switch --trace-json "$RESULT_DIR/${stem}_trace.json")
    fi
    local scheduler_args=()
    if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
        scheduler_args=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi
    PHASE_SWITCH_SYNC_COPY="$sync_copy" "$PYTHON_BIN" "$ARTIFACT_DIR/benchmark_generate.py" \
        --model "$model" \
        --label "$label" \
        --scenario "$SCENARIO" \
        --input-len "$INPUT_LEN" \
        --output-len "$out_len" \
        --batch-size "$BATCH_SIZE" \
        --output-json "$RESULT_DIR/${stem}.json" \
        --repo-root "$REPO_ROOT" \
        --artifact-dir "$ARTIFACT_DIR" \
        --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
        "${scheduler_args[@]}" \
        "${trace_args[@]}" \
        > "$LOG_DIR/${stem}.log" 2>&1
}

run_pair() {
    local label="$1"
    local model="$2"
    local phase_flag="$3"
    local sync_copy="$4"
    run_one "$label" "$model" 1 "$phase_flag" "$sync_copy"
    run_one "$label" "$model" "$OUTPUT_LEN" "$phase_flag" "$sync_copy"
}

run_pair dense_nvfp4 "$DENSE_NVFP4_MODEL" 0 0
run_pair w4a16_ours "$W4A16_MODEL" 0 0
run_pair phase_switch_overlap "$PHASE_MODEL" 1 0
run_pair phase_switch_sync "$PHASE_MODEL" 1 1

"$PYTHON_BIN" "$ARTIFACT_DIR/summarize_decomposed_benchmark.py" \
    --result-dir "$RESULT_DIR" \
    --scenario "$SCENARIO" \
    --output-len "$OUTPUT_LEN" \
    --output-json "$RESULT_DIR/summary.json" \
    --output-md "$RESULT_DIR/summary.md"

echo "wrote $RESULT_DIR/summary.json"
echo "wrote $RESULT_DIR/summary.md"
