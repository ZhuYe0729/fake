# FakeVLM Prefill Global Pareto Summary

## Uniform Accuracy Baselines

| Method | Accuracy |
|---|---:|
| `dense_bf16` | 0.986400 |
| `sparse_bf16` | 0.985200 |
| `dense_nvfp4` | 0.987000 |
| `sparse_nvfp4` | 0.768600 |
| `marlin_weight_only` | 0.987600 |
| `dense_nvfp4_prefill_marlin_decode` | 0.986800 |

## Validated Pareto Points

| Batch | Point | Pred latency ms | Pred quality cost | E2E ms | Accuracy | Counts |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 0 | 4.883 | 0 | 1294.259949 | 1.00000000 | dense_bf16=2 |
| 16 | 3 | 2.516 | 2.83271e-06 | 1292.545349 | 1.00000000 | sparse_bf16=2 |
