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
| 16 | 0 | 984.679 | 0 | 1313.568433 | 0.98640000 | dense_bf16=224 |
| 16 | 4 | 968.954 | 0.000187812 | 1297.143689 | 0.98680000 | dense_bf16=220, dense_nvfp4=4 |
| 16 | 8 | 941.435 | 0.000966092 | 1273.648804 | 0.98700000 | dense_bf16=213, dense_nvfp4=11 |
| 16 | 11 | 894.259 | 0.00240504 | 1232.168921 | 0.98640000 | dense_bf16=201, dense_nvfp4=23 |
| 16 | 15 | 752.733 | 0.00628876 | 1110.211169 | 0.98660000 | dense_bf16=165, dense_nvfp4=59 |
| 16 | 18 | 587.804 | 0.0230503 | 957.340991 | 0.98480000 | dense_bf16=103, dense_nvfp4=64, sparse_bf16=57 |
| 16 | 19 | 548.753 | 0.0318137 | 917.858167 | 0.98640000 | dense_bf16=70, dense_nvfp4=64, sparse_bf16=90 |
| 16 | 20 | 500.235 | 0.0421374 | 866.804022 | 0.98420000 | dense_bf16=29, dense_nvfp4=64, sparse_bf16=131 |
| 16 | 21 | 460.515 | 0.0627763 | 829.793335 | 0.97900000 | dense_nvfp4=53, sparse_bf16=160, sparse_nvfp4=11 |
| 16 | 22 | 451.184 | 0.0998303 | 819.160175 | 0.94840000 | dense_nvfp4=34, sparse_bf16=160, sparse_nvfp4=30 |
| 16 | 25 | 434.485 | 0.23573 | 803.449835 | 0.95280000 | sparse_bf16=160, sparse_nvfp4=64 |
