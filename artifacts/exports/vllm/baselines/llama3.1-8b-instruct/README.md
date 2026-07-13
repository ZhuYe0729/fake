# Llama-3.1-8B-Instruct vLLM Baselines

This directory contains baseline scripts for Llama-3.1-8B-Instruct:

- Dense BF16 model: `/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- Uniform compressed methods: `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, `marlin_nvfp4`
- Speed scenarios:
  - `prefill_only`: `batch=8,input_seq=2048,output_seq=1`
  - `prefill_decode`: `batch=16,input_seq=2048,output_seq=80`
- PMPD-style quality datasets:
  - `cnn_dm_1000`: fixed 1000-example CNN/DailyMail subset, not the full CNN/DM test set
  - `dsum`
  - `IWSLT`

Compression is real calibrated compression. Run `scripts/prepare_uniform_compressed.py`
before export; do not use simple module replacement as a baseline result.

Quality uses PMPD's Claude-style `Human`/`Assistant` prompt rather than the
Llama-3.1 native chat template. This is intentional: it matches the Llama2
baseline protocol for cross-model comparison. It is not a measurement of
native-chat-template deployment quality.

## Recommended Run Order

```bash
# In the project environment with compression dependencies.
python artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/prepare_uniform_compressed.py \
  --methods sparse_bf16,dense_nvfp4,sparse_nvfp4,marlin_nvfp4 \
  --gpu 0

# In the environment with vLLM and CUDA kernels.
python artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/export_uniform_vllm.py \
  --methods dense_nvfp4,sparse_bf16,sparse_nvfp4,marlin_nvfp4

bash artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/run_all_speed.sh
bash artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/run_all_quality.sh

python artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/summarize_results.py
```

`run_all_quality.sh` enables `--resume`. The evaluator validates the existing
run configuration and JSONL prefix, skips completed questions, and truncates
only a malformed final partial line. Run without `--resume` to intentionally
overwrite a dataset result.

## Outputs

- `prepared/{method}/`: calibrated compressed dense-state artifacts.
- `checkpoints/uniform_{method}/`: fused vLLM compressed checkpoints.
- `results/speed/`: latency and throughput CSV/JSON files.
- `results/quality/`: PMPD-style predictions and metrics.
- `results/summary/`: joined summary tables.

The speed script reports end-to-end batch latency and throughput. It also records
TTFT/TPOT columns. By default those are reliable offline-derived values:
TTFT is measured with the same batch/input and `output_seq=1`; TPOT for decode
is derived from `(decode_e2e - ttft) / (output_seq - 1)`.
