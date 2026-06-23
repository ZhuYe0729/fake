# Qwen/Llama Cross-Model Robustness

This debug experiment measures Qwen3.5 and Llama speed across model sizes and workloads.

Default models:

- `0.8B`
- `2B`
- `4B`
- `9B`
- `llama2-7b`
- `llama31-8b`

Default workloads:

| Scenario | Batch | Input tokens | Output tokens |
|---|---:|---:|---:|
| `prefill_only` | 16 | 1024 | 0 |
| `normal_01` | 1 | 16384 | 32 |
| `normal_02` | 1 | 16384 | 256 |

Methods:

- `dense_bf16`
- `uniform_dense_nvfp4`
- `uniform_sparse_bf16`
- `uniform_sparse_nvfp4`
- `uniform_marlin_weight_only`
- `uniform_dense_nvfp4_prefill_marlin_decode`
- `our_linear_hybrid`

Smoke:

```bash
CONDA_ENV=cospaq MODELS="0.8B" SCENARIOS="prefill_only" METHODS="dense_bf16 our_linear_hybrid" WARMUP=1 ITERS=2 \
  bash artifacts/debug/027_qwen_cross_model_robustness/scripts/launch_cross_model_4gpu.sh
```

Full:

```bash
CONDA_ENV=cospaq GPUS="5 6 7" bash artifacts/debug/027_qwen_cross_model_robustness/scripts/launch_cross_model_4gpu.sh
```

Run only Llama additions:

```bash
CONDA_ENV=cospaq GPUS="5 6 7" MODELS="llama2-7b llama31-8b" \
  bash artifacts/debug/027_qwen_cross_model_robustness/scripts/launch_cross_model_4gpu.sh
```

Summarize:

```bash
python artifacts/debug/027_qwen_cross_model_robustness/scripts/summarize_cross_model.py
```
