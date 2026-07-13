# Llama2-7B vLLM Baseline

This directory is the fixed vLLM baseline for Llama2-7B uniform compressed models.

## Exported Models

- `uniform_dense_nvfp4`
- `uniform_sparse_bf16`
- `uniform_sparse_nvfp4`

The original dense BF16 baseline uses `/root/wja/data/models/LLM-Research/llama-2-7b`.

## Speed Benchmark

Run from the repository root:

```bash
source /home/agent/wja/miniconda3/etc/profile.d/conda.sh
conda activate vllm
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/wja/project/my/cospaq/fake/fake/kernels/cutlass/cutlass_wrapper:$PYTHONPATH \
  python artifacts/exports/vllm/llama2_7b_018/scripts/benchmark_prefill_vllm.py
```

The benchmark scenario is `prefill_plus_1_decode`: batch size 16, prompt length 1024, `max_tokens=1`, `detokenize=False`, prefix caching disabled. This is the closest stable vLLM generate API approximation to prefill-only.

## Quality Baseline

Quality uses the existing 018 full ARC-Challenge results with 1172 examples, not the limit-128 validation.

## Baseline Results

Primary fixed summary:

- `summary/uniform_vllm_prefill_speed_quality_baseline.csv`
- `summary/uniform_vllm_prefill_speed_quality_baseline.md`

Current vLLM eager baseline:

| method | median ms | speedup | tok/s | NLL delta | ARC-C acc_norm |
|---|---:|---:|---:|---:|---:|
| dense_bf16 | 1051.619 | 1.000 | 15579.8 | 0.0000 | 0.4514 |
| dense_nvfp4 | 563.644 | 1.866 | 29068.0 | 0.0820 | 0.4377 |
| sparse_bf16 | 631.798 | 1.664 | 25932.4 | 0.3503 | 0.3379 |
| sparse_nvfp4 | 504.653 | 2.084 | 32465.9 | 1.3184 | 0.2287 |
