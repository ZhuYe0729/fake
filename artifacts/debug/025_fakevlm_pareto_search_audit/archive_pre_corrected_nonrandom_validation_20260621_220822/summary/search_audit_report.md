# FakeVLM Pareto Search Audit

- Total policies validated: 68
- Search policies validated: 60
- Non-dominated searched policies: 5
- 024 reference points: 8
- 024 reference source: measured with 025 validator
- Reference points dominated by searched policies: 6
- Max latency improvement at equal/higher accuracy: 30.49%

## Non-dominated searched policies

| key | family | accuracy | e2e ms | counts |
|---|---|---:|---:|---|
| `neighborhood_neighborhood_016` | neighborhood | 0.990000 | 906.680 | dense_bf16=50, dense_nvfp4=69, sparse_bf16=99, sparse_nvfp4=6 |
| `neighborhood_neighborhood_019` | neighborhood | 0.993000 | 911.156 | dense_bf16=48, dense_nvfp4=70, sparse_bf16=89, sparse_nvfp4=17 |
| `neighborhood_neighborhood_013` | neighborhood | 0.995000 | 1041.629 | dense_bf16=126, dense_nvfp4=71, sparse_bf16=16, sparse_nvfp4=11 |
| `random_random_020` | random | 0.996000 | 1154.348 | dense_bf16=141, dense_nvfp4=21, sparse_bf16=28, sparse_nvfp4=34 |
| `random_random_004` | random | 0.997000 | 1180.429 | dense_bf16=154, dense_nvfp4=26, sparse_bf16=28, sparse_nvfp4=16 |

## Gap to 024 reference

| reference | search key | improvement |
|---|---|---:|
| batch_16_point_000 | `neighborhood_neighborhood_019` | 30.49% |
| batch_16_point_005 | `neighborhood_neighborhood_013` | 19.13% |
| batch_16_point_009 | `neighborhood_neighborhood_013` | 19.20% |
| batch_16_point_013 | `neighborhood_neighborhood_013` | 16.26% |
| batch_16_point_018 | `neighborhood_neighborhood_013` | 10.90% |
| batch_16_point_022 | `neighborhood_neighborhood_013` | 2.36% |
| batch_16_point_026 | `` | 0.00% |
| batch_16_point_030 | `` | 0.00% |
