#!/usr/bin/env bash
# Materialize one canonical policy and run the formal B=8/O=64 speed protocol.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 POLICY_ID POLICY_JSON GPU" >&2
  exit 2
fi
POLICY_ID="$1"
POLICY_JSON="$2"
GPU="$3"
ROOT="${COSPAQ_EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${COSPAQ_REPO_ROOT:-/root/wja/project/my/cospaq/fake}"
PYTHON="${VLLM_PYTHON:-/home/agent/wja/miniconda3/envs/vllm/bin/python}"
EXPORTER="${COSPAQ_EXPORTER:-$REPO/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py}"
BENCH="$REPO/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"
OUT="${COSPAQ_EXPERIMENT_DIR:-$ROOT/llama2_7b_chat}/speed/runs/$POLICY_ID"
CHECKPOINT="$OUT/checkpoint"
CANONICAL_DIR="${COSPAQ_CANONICAL_DIR:-$REPO/artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/canonical/prepared}"
VERIFY_CHECKPOINT="${COSPAQ_VERIFY_CHECKPOINT:-$ROOT/scripts/verify_canonical_checkpoint.py}"

mkdir -p "$OUT"
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$EXPORTER" --policy-json "$POLICY_JSON" \
    --output-dir "$CHECKPOINT" --model-path "${COSPAQ_MODEL_PATH:-/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf}" \
    --canonical-sparse-bf16-state "$CANONICAL_DIR/sparse_bf16/model.pt" \
    --canonical-sparse-nvfp4-state "$CANONICAL_DIR/sparse_nvfp4/model.pt" --force
fi
"$PYTHON" "$VERIFY_CHECKPOINT" \
  --policy "$POLICY_JSON" --checkpoint "$CHECKPOINT"

export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=0 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$BENCH" --checkpoint "$CHECKPOINT" \
  --scenario prefill_decode --batch 8 --input-seq 2048 --output-seq 64 \
  --output-dir "$OUT" --gpu-memory-utilization 0.80 --warmup-iters 1 --iters 5
