#!/usr/bin/env bash
# Fresh, deliberately small closure for one representative policy.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 POLICY_ID GPU" >&2
  exit 2
fi
POLICY_ID="$1"
GPU="$2"
: "${COSPAQ_REPO_ROOT:?source config.env first}"
: "${COSPAQ_VLLM_ROOT:?source config.env first}"
: "${COSPAQ_MODEL_PATH:?source config.env first}"
: "${COSPAQ_RUN_ROOT:?source config.env first}"
: "${COSPAQ_CANONICAL_DIR:?source config.env first}"
: "${VLLM_PYTHON:?source config.env first}"

POLICY="$COSPAQ_RUN_ROOT/prefill_decode/policies/prefill_decode/$POLICY_ID.json"
OUT="$COSPAQ_RUN_ROOT/validation/smoke/$POLICY_ID"
CHECKPOINT="$OUT/checkpoint"
CUTLASS="$COSPAQ_REPO_ROOT/fake/kernels/cutlass/cutlass_wrapper"
EXPORTER="$COSPAQ_REPO_ROOT/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/export_max_speed_checkpoint.py"
VERIFY="$COSPAQ_REPO_ROOT/artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/verify_canonical_checkpoint.py"
PREFILL_NLL="$COSPAQ_REPO_ROOT/artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_runtime_prefill_nll.py"
DECODE_NLL="$COSPAQ_REPO_ROOT/artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py"
BENCH="$COSPAQ_REPO_ROOT/artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py"

test -f "$POLICY"
test -f "$COSPAQ_CANONICAL_DIR/sparse_bf16/model.pt"
test -f "$COSPAQ_CANONICAL_DIR/sparse_nvfp4/model.pt"
mkdir -p "$OUT" "${COSPAQ_EXT_CACHE_ROOT:-$OUT/extensions}/$POLICY_ID"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$COSPAQ_VLLM_ROOT/vllm:$COSPAQ_VLLM_ROOT:$CUTLASS${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${COSPAQ_EXT_CACHE_ROOT:-$OUT/extensions}/$POLICY_ID"
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export PHASE_HETERO_TRACE=1 PHASE_HETERO_GC_EVERY_APPLY=1 PHASE_HETERO_GC_DECODE=0
export PHASE_HETERO_WAIT_ONCE=1 PHASE_HETERO_RELEASE_PREFILL=1

if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
  "$VLLM_PYTHON" "$EXPORTER" \
    --policy-json "$POLICY" --model-path "$COSPAQ_MODEL_PATH" \
    --output-dir "$CHECKPOINT" --cutlass-wrapper-path "$CUTLASS" \
    --canonical-sparse-bf16-state "$COSPAQ_CANONICAL_DIR/sparse_bf16/model.pt" \
    --canonical-sparse-nvfp4-state "$COSPAQ_CANONICAL_DIR/sparse_nvfp4/model.pt"
fi
"$VLLM_PYTHON" "$VERIFY" --policy "$POLICY" --checkpoint "$CHECKPOINT"

"$VLLM_PYTHON" "$PREFILL_NLL" \
  --checkpoint "$CHECKPOINT" --tokenizer "$COSPAQ_MODEL_PATH" \
  --samples "$COSPAQ_RUN_ROOT/prefill_only/samples/wikitext_2048_targets.pt" \
  --output "$OUT/prefill_only_nll.json" --label "$POLICY_ID" \
  --policy-json "$POLICY" --phase-hetero --blocks 2

"$VLLM_PYTHON" "$DECODE_NLL" \
  --model-path "$COSPAQ_MODEL_PATH" --tokenizer "$COSPAQ_MODEL_PATH" \
  --policy-json "$POLICY" \
  --samples "$COSPAQ_RUN_ROOT/prefill_decode/samples/wikitext_2048_64.pt" \
  --output "$OUT/prefill_decode_nll.json" --label "$POLICY_ID" --blocks 2 \
  --canonical-sparse-bf16-state "$COSPAQ_CANONICAL_DIR/sparse_bf16/model.pt" \
  --canonical-sparse-nvfp4-state "$COSPAQ_CANONICAL_DIR/sparse_nvfp4/model.pt" \
  --input-tokens 2048 --output-tokens 64 --batch-size 8 --gpu-memory-utilization 0.80

"$VLLM_PYTHON" "$BENCH" \
  --checkpoint "$CHECKPOINT" --scenario prefill_decode --batch 8 \
  --input-seq 2048 --output-seq 64 --output-dir "$OUT/speed_raw" \
  --vllm-root "$COSPAQ_VLLM_ROOT" --cutlass-wrapper-path "$CUTLASS" \
  --gpu-memory-utilization 0.80 --warmup-iters 1 --iters 2

"$VLLM_PYTHON" -c 'import csv,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=list(csv.DictReader((p/"summary.csv").open())); pathlib.Path(sys.argv[2]).write_text(json.dumps({"source":str(p),"summary":rows},indent=2)+"\n")' "$OUT/speed_raw" "$OUT/speed.json"
