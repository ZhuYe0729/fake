# End-to-end paper reproduction

This runbook reproduces the final paper protocol without relying on the old
numbered debug directories. Commands are shown for Llama-2-7B-Chat; use the
corresponding Llama-3.1 config and model path for the second model.

## 1. Install and configure

Use the project environment for calibration/modeling and the patched vLLM
environment for runtime commands. Install DASO in both when practical:

```bash
cd /path/to/daso
python -m pip install -e '.[eval,plot]'

export LLAMA2_7B_CHAT_PATH=/models/Llama-2-7b-chat-hf
export LLAMA31_8B_INSTRUCT_PATH=/models/Meta-Llama-3.1-8B-Instruct
export DASO_VLLM_ROOT=/src/patched-vllm
export DASO_CUTLASS_WRAPPER_ROOT=/src/cutlass_wrapper
export DASO_RUN_ROOT=/experiments/daso
```

Copy the required example config to a writable run directory. Never edit the
frozen examples to contain machine-local paths.

```bash
python workflows/paper/preflight.py \
  --config configs/experiments/llama2_prefill_only.example.toml \
  --output "$DASO_RUN_ROOT/llama2_7b_chat/prefill_only/validation/preflight.json"
```

Preflight must report `ok: true`. Serially prewarm all four compressed runtime
extensions before launching concurrent quality jobs; never launch several
first-time JIT builds against the same extension directory.

## 2. Freeze calibration samples and canonical sparse states

Create one deterministic WikiText tensor and reuse it across methods:

```bash
python workflows/paper/prepare_wikitext_samples.py \
  --tokenizer "$LLAMA2_7B_CHAT_PATH" \
  --dataset-arrow /datasets/wikitext/train.arrow \
  --samples 128 --sequence-length 2112 --seed 0 \
  --output "$DASO_RUN_ROOT/llama2_7b_chat/samples/wikitext_2048_64.pt"

CUDA_VISIBLE_DEVICES=0 python workflows/paper/prepare_canonical_sparse.py \
  --model-path "$LLAMA2_7B_CHAT_PATH" \
  --samples "$DASO_RUN_ROOT/llama2_7b_chat/samples/wikitext_2048_64.pt" \
  --output-root "$DASO_RUN_ROOT/llama2_7b_chat/canonical"
```

The output contains calibrated `sparse_bf16` and pre-quantization
`sparse_nvfp4` BF16 states. The phase exporter performs the sole final NVFP4
quantization/pack. Do not pass `--prune` and do not quantize the canonical state
twice.

## 3. Collect quality features and real-runtime labels

Frozen 72-policy calibration sets are under
`results/paper/modeling_calibration/<model>/<scenario>/policies`. Copy them to
the run tree without changing policy JSON content:

```bash
mkdir -p "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/policies"
cp results/paper/modeling_calibration/llama2_7b_chat/prefill_decode/policies/* \
  "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/policies/"
```

Collect per-layer/module local relative-MSE with the same CUTLASS wrappers used
by phase-vLLM. For prefill-decode, collect separate `prefill_*.csv` and
`decode_*.csv`; dense BF16 is the implicit zero-error reference:

```bash
for phase in prefill decode; do
  for method in dense_nvfp4 sparse_bf16 sparse_nvfp4 w4a16_ours; do
    CUDA_VISIBLE_DEVICES=0 python workflows/paper/collect_local_errors.py \
      --model-path "$LLAMA2_7B_CHAT_PATH" \
      --samples "$DASO_RUN_ROOT/llama2_7b_chat/samples/wikitext_2048_64.pt" \
      --canonical-root "$DASO_RUN_ROOT/llama2_7b_chat/canonical" \
      --cutlass-wrapper-path "$DASO_CUTLASS_WRAPPER_ROOT" \
      --phase "$phase" --method "$method" --input-tokens 2048 \
      --output-tokens 64 --blocks 16 \
      --output "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/local_errors/${phase}_${method}.csv"
  done
done
```

Export every policy through the external phase exporter, using both canonical
states. Run real vLLM NLL:

```bash
python workflows/paper/runtime/evaluate_prefill_nll.py \
  --checkpoint /runs/checkpoint --tokenizer "$LLAMA2_7B_CHAT_PATH" \
  --samples "$DASO_RUN_ROOT/llama2_7b_chat/samples/wikitext_2048_64.pt" \
  --output /runs/p00_prefill.json --label p00 --phase-hetero \
  --policy-json /runs/p00.json --blocks 100 \
  --vllm-root "$DASO_VLLM_ROOT" \
  --cutlass-wrapper-path "$DASO_CUTLASS_WRAPPER_ROOT"

python workflows/paper/runtime/run_policy_nll.py \
  --model-path "$LLAMA2_7B_CHAT_PATH" --tokenizer "$LLAMA2_7B_CHAT_PATH" \
  --policy-json /runs/p00.json \
  --samples "$DASO_RUN_ROOT/llama2_7b_chat/samples/wikitext_2048_64.pt" \
  --output /runs/p00_decode.json --label p00 --blocks 100 \
  --input-tokens 2048 --output-tokens 64 --batch-size 8 \
  --canonical-sparse-bf16-state "$DASO_RUN_ROOT/llama2_7b_chat/canonical/sparse_bf16/model.pt" \
  --canonical-sparse-nvfp4-state "$DASO_RUN_ROOT/llama2_7b_chat/canonical/sparse_nvfp4/model.pt" \
  --vllm-root "$DASO_VLLM_ROOT" \
  --cutlass-wrapper-path "$DASO_CUTLASS_WRAPPER_ROOT"
```

Quality jobs may run on separate GPUs after extension prewarm. A policy is
complete only when result, export provenance, policy hash, sample hash, and
phase trace exist. Merge the 72 averages into `prefill_only.csv` or
`prefill_decode.csv` with `target_delta_nll` relative to the same-runtime BF16
policy:

```bash
python workflows/paper/merge_nll.py \
  --manifest "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/policies/manifest.json" \
  --raw-dir "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/nll/raw" \
  --reference-policy p00 --require-canonical-provenance \
  --output "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/nll/prefill_decode.csv"
```

Fit the final structure/phase-aware model:

```bash
python workflows/paper/fit_quality.py \
  --manifest /runs/policies/manifest.json --policy-dir /runs/policies \
  --local-errors /runs/local_errors --nll /runs/prefill_decode.csv \
  --scenario prefill_decode --output /runs/reports/quality
```

## 4. Build latency actions and solve

Run the external `KernelLatencyPredictor` profile for the target GPU and all
Llama module shapes. Store a row per phase/module/kernel with support and
latency. Do not reuse RTX 5090 calibration on RTX PRO 6000:

```bash
python workflows/paper/profile_latency_actions.py \
  --model-path "$LLAMA2_7B_CHAT_PATH" \
  --cutlass-wrapper-path "$DASO_CUTLASS_WRAPPER_ROOT" \
  --scenario prefill_decode --batch 8 --input-tokens 2048 --output-tokens 64 \
  --output "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/speed/action_support.csv"
```

Measure a small fixed design of diverse policies with the real speed runner.
Write `policy_id,policy_json,measured_ms,split` to `speed/design.csv`, where
`measured_ms` is the median of five fresh-process samples. Fit the monotone E2E
correction and keep the holdout metrics:

```bash
python workflows/paper/fit_speed_calibration.py \
  --design "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/speed/design.csv" \
  --latency-actions "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/speed/action_support.csv" \
  --scenario prefill_decode --decode-steps 63 \
  --output "$DASO_RUN_ROOT/llama2_7b_chat/prefill_decode/speed/calibration"
```

The local Roofline totals determine action ordering in the optimizer; the
monotone fit calibrates those totals to E2E time without changing their order.
All paper points are subsequently closed with measured E2E latency.

```bash
python workflows/paper/build_action_table.py \
  --quality-model /runs/reports/quality/model.json \
  --local-errors /runs/local_errors \
  --latency-actions /runs/speed/action_support.csv \
  --scenario prefill_decode --decode-steps 63 \
  --output /runs/solver/actions.csv

python workflows/paper/solve_from_actions.py \
  --actions /runs/solver/actions.csv --scenario prefill_decode \
  --output /runs/pareto
```

The predictor screens candidates; paper speed and quality always come from the
real runtime closure. Validate dense, near-lossless, bridge, balanced, and
max-speed points before expanding the sweep.

## 5. Real speed and downstream quality

Measure uniform and DASO policies with the same exported phase runtime:

```bash
python workflows/paper/runtime/benchmark_phase_hetero.py \
  --checkpoint /runs/checkpoint --scenario prefill_decode \
  --batch 8 --input-seq 2048 --output-seq 64 \
  --output-dir /runs/speed/point_005 --warmup-iters 1 --iters 5 \
  --gpu-memory-utilization 0.80 --reuse-llm \
  --vllm-root "$DASO_VLLM_ROOT" \
  --cutlass-wrapper-path "$DASO_CUTLASS_WRAPPER_ROOT"
```

Formal speed runs are serial and require an otherwise idle GPU. Quality runs
can be sharded across GPUs. Prefill-only uses ARC-Challenge `acc_norm`,
ARC-Easy `acc`, WinoGrande `acc`, MMLU `acc`, and WikiText PPL. Generation uses
legacy/common PMPD prompts, CNN/DailyMail and DialogSum ROUGE-L, and IWSLT'17
SacreBLEU. Keep BF16 KV cache, disable prefix cache and chunked prefill, and use
the same EOS/EOT policy for all methods.

Generation shards are intentionally produced with `--skip-metrics`. Merge the
non-overlapping JSONL shards in question-id order, reject duplicate IDs, and
score the merged file once:

```bash
python workflows/paper/runtime/merge_generation_shards.py \
  /runs/tasks/point_005/cnn_dm/shards/*.jsonl \
  --expected-count 1000 \
  --output /runs/tasks/point_005/cnn_dm/point_005-fp16.jsonl

python workflows/paper/runtime/score_generation.py \
  --vllm-root "$DASO_VLLM_ROOT" \
  --answers /runs/tasks/point_005/cnn_dm/point_005-fp16.jsonl \
  --dataset cnn_dm --bertscore-model /models/roberta-large \
  --output /runs/tasks/point_005/cnn_dm/metrics.json
```

## 6. Paper and appendix artifacts

```bash
MPLCONFIGDIR=/tmp/daso-mpl bash workflows/paper/rebuild_paper_artifacts.sh
```

This rebuilds all data-driven figures, refreshes hashes, and validates required
tables/results against the final paper. The appendix audit re-evaluates 15
random, 10 neighborhood, and five leverage-guided challengers with the exact
same real-runtime speed and downstream protocols as DASO points.
