#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ARTIFACT_DIR/results_pmpd_chat_cnn100}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_DIR/logs_pmpd_chat_cnn100}"

PYTHON_BIN="${PYTHON_BIN:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
CUTLASS_WRAPPER="${CUTLASS_WRAPPER:-/home/agent/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper}"
QUESTION_END="${QUESTION_END:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"

BF16_MODEL="${BF16_MODEL:-/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf}"
DENSE_NVFP4_MODEL="${DENSE_NVFP4_MODEL:-$ARTIFACT_DIR/models_chat/llama-2-7b-chat-nvfp4-mytest}"
W4A16_MODEL="${W4A16_MODEL:-$ARTIFACT_DIR/models_chat/llama-2-7b-chat-w4a16-ours}"
PHASE_MODEL="${PHASE_MODEL:-$ARTIFACT_DIR/models_chat/llama-2-7b-chat-phase-switch-dense-nvfp4-marlin}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$REPO_ROOT/vllm:$REPO_ROOT:$ARTIFACT_DIR:$CUTLASS_WRAPPER${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PHASE_SWITCH_TRACE="${PHASE_SWITCH_TRACE:-0}"
export PHASE_SWITCH_RELEASE_PREFILL="${PHASE_SWITCH_RELEASE_PREFILL:-0}"

run_one() {
    local label="$1"
    local model="$2"
    local phase="$3"
    local extra=()
    if [[ "$phase" == "1" ]]; then
        extra=(--phase-switch)
    fi
    "$PYTHON_BIN" "$ARTIFACT_DIR/pmpd_vllm_eval.py" \
        --dataset cnn_dm_1000 \
        --question-end "$QUESTION_END" \
        --batch-size "$BATCH_SIZE" \
        --model-path "$model" \
        --model-id "$label" \
        --output-dir "$OUT_DIR" \
        --repo-root "$REPO_ROOT" \
        --artifact-dir "$ARTIFACT_DIR" \
        --cutlass-wrapper-path "$CUTLASS_WRAPPER" \
        --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large \
        --skip-metrics \
        "${extra[@]}" \
        > "$LOG_DIR/${label}.log" 2>&1

    "$PYTHON_BIN" pmpd_eval_kit/pmpd_eval.py \
        --dataset cnn_dm_1000 \
        --metrics-only "$OUT_DIR/cnn_dm_1000/${label}-fp16.jsonl" \
        --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large \
        > "$LOG_DIR/${label}_metrics.log" 2>&1
    cp "$OUT_DIR/cnn_dm_1000/metrics.json" \
       "$OUT_DIR/cnn_dm_1000/${label}_metrics.json"
}

run_one dense_bf16 "$BF16_MODEL" 0
run_one dense_nvfp4 "$DENSE_NVFP4_MODEL" 0
run_one w4a16 "$W4A16_MODEL" 0
run_one dense_nvfp4_plus_w4a16 "$PHASE_MODEL" 1

"$PYTHON_BIN" "$ARTIFACT_DIR/summarize_pmpd_speed_quality.py" \
    --pmpd-dir "$OUT_DIR/cnn_dm_1000" \
    --speed-summary "$ARTIFACT_DIR/results_end2end_notrace/summary.json" \
    --output-json "$OUT_DIR/summary_speed_quality.json" \
    --output-md "$OUT_DIR/summary_speed_quality.md"

echo "wrote $OUT_DIR/summary_speed_quality.json"
echo "wrote $OUT_DIR/summary_speed_quality.md"
