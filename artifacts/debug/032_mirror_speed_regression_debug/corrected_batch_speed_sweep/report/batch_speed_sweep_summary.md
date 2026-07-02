# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 1 | dense_default_amp | dense_default_amp | 39.917 | 1.000x |
| 1 | uniform_dense_bf16 | uniform_dense_bf16 | 26.995 | 1.479x |
| 1 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 69.693 | 0.573x |
| 1 | uniform_sparse_bf16 | uniform_sparse_bf16 | 46.088 | 0.866x |
| 1 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 83.474 | 0.478x |
| 1 | ours_best | ours_gate_up_sparse_bf16_64 | 34.240 | 1.166x |
| 2 | dense_default_amp | dense_default_amp | 40.381 | 1.000x |
| 2 | uniform_dense_bf16 | uniform_dense_bf16 | 29.062 | 1.389x |
| 2 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 64.347 | 0.628x |
| 2 | uniform_sparse_bf16 | uniform_sparse_bf16 | 50.032 | 0.807x |
| 2 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 83.472 | 0.484x |
| 2 | ours_best | ours_gate_up_sparse_bf16_64 | 35.980 | 1.122x |
| 4 | dense_default_amp | dense_default_amp | 38.906 | 1.000x |
| 4 | uniform_dense_bf16 | uniform_dense_bf16 | 29.033 | 1.340x |
| 4 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 76.150 | 0.511x |
| 4 | uniform_sparse_bf16 | uniform_sparse_bf16 | 43.989 | 0.884x |
| 4 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 80.991 | 0.480x |
| 4 | ours_best | ours_gate_up_sparse_bf16_64 | 38.203 | 1.018x |
| 8 | dense_default_amp | dense_default_amp | 40.318 | 1.000x |
| 8 | uniform_dense_bf16 | uniform_dense_bf16 | 29.852 | 1.351x |
| 8 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 82.923 | 0.486x |
| 8 | uniform_sparse_bf16 | uniform_sparse_bf16 | 40.000 | 1.008x |
| 8 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 86.434 | 0.466x |
| 8 | ours_best | ours_gate_up_sparse_bf16_64 | 35.558 | 1.134x |
| 16 | dense_default_amp | dense_default_amp | 56.014 | 1.000x |
| 16 | uniform_dense_bf16 | uniform_dense_bf16 | 51.751 | 1.082x |
| 16 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 86.207 | 0.650x |
| 16 | uniform_sparse_bf16 | uniform_sparse_bf16 | 45.388 | 1.234x |
| 16 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 91.897 | 0.610x |
| 16 | ours_best | ours_mlp_sparse_bf16_96 | 43.076 | 1.300x |
| 32 | dense_default_amp | dense_default_amp | 103.918 | 1.000x |
| 32 | uniform_dense_bf16 | uniform_dense_bf16 | 94.319 | 1.102x |
| 32 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 119.374 | 0.871x |
| 32 | uniform_sparse_bf16 | uniform_sparse_bf16 | 71.935 | 1.445x |
| 32 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 117.138 | 0.887x |
| 32 | ours_best | ours_extreme_fastest_microbench | 74.277 | 1.399x |
