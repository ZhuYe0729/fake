# FakeVLM Pareto Search Audit

- Total policies validated: 71
- Search policies validated: 60
- Non-dominated searched policies: 10
- 024 reference points: 11
- 024 reference source: measured with 025 validator
- Reference points dominated by searched policies: 4
- Max latency improvement at equal/higher accuracy: 25.78%

## Non-dominated searched policies

| key | family | accuracy | e2e ms | counts |
|---|---|---:|---:|---|
| `neighborhood_neighborhood_016` | neighborhood | 0.938000 | 824.239 | dense_bf16=2, dense_nvfp4=3, sparse_bf16=152, sparse_nvfp4=67 |
| `neighborhood_neighborhood_017` | neighborhood | 0.943000 | 829.134 | dense_bf16=3, dense_nvfp4=13, sparse_bf16=143, sparse_nvfp4=65 |
| `suspicious_suspicious_007` | suspicious | 0.981000 | 836.083 | dense_bf16=6, dense_nvfp4=8, sparse_bf16=151, sparse_nvfp4=59 |
| `random_random_016` | random | 0.986000 | 920.208 | dense_bf16=0, dense_nvfp4=84, sparse_bf16=65, sparse_nvfp4=75 |
| `random_random_008` | random | 0.988000 | 920.437 | dense_bf16=0, dense_nvfp4=66, sparse_bf16=88, sparse_nvfp4=70 |
| `neighborhood_neighborhood_012` | neighborhood | 0.991000 | 957.979 | dense_bf16=94, dense_nvfp4=66, sparse_bf16=56, sparse_nvfp4=8 |
| `neighborhood_neighborhood_014` | neighborhood | 0.992000 | 960.037 | dense_bf16=92, dense_nvfp4=58, sparse_bf16=58, sparse_nvfp4=16 |
| `suspicious_suspicious_003` | suspicious | 0.993000 | 973.946 | dense_bf16=101, dense_nvfp4=57, sparse_bf16=60, sparse_nvfp4=6 |
| `random_random_020` | random | 0.996000 | 1154.348 | dense_bf16=141, dense_nvfp4=21, sparse_bf16=28, sparse_nvfp4=34 |
| `random_random_004` | random | 0.997000 | 1180.429 | dense_bf16=154, dense_nvfp4=26, sparse_bf16=28, sparse_nvfp4=16 |

## Gap to 024 reference

| reference | search key | improvement |
|---|---|---:|
| batch_16_point_000 | `suspicious_suspicious_003` | 25.78% |
| batch_16_point_004 | `suspicious_suspicious_003` | 24.58% |
| batch_16_point_008 | `random_random_020` | 9.96% |
| batch_16_point_011 | `suspicious_suspicious_003` | 20.22% |
| batch_16_point_015 | `` | 0.00% |
| batch_16_point_018 | `` | 0.00% |
| batch_16_point_019 | `` | 0.00% |
| batch_16_point_020 | `` | 0.00% |
| batch_16_point_021 | `` | 0.00% |
| batch_16_point_022 | `` | 0.00% |
| batch_16_point_025 | `` | 0.00% |
