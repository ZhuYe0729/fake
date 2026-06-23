# FakeVLM Cross-Workload Robustness

This debug experiment measures FakeVLM speed across `prefill_only`, `normal_01`, and `normal_02`.

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

Run smoke:

```bash
CONDA_ENV=cospaq SAMPLE_LIMIT=2 WARMUP=1 ITERS=2 METHODS="dense_bf16 uniform_marlin_weight_only our_linear_hybrid" \
  bash artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/launch_full_4gpu.sh
```

Run full:

```bash
CONDA_ENV=cospaq bash artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/launch_full_4gpu.sh
```

Summarize:

```bash
python artifacts/debug/026_fakevlm_cross_workload_robustness/scripts/summarize_cross_workload.py
```
