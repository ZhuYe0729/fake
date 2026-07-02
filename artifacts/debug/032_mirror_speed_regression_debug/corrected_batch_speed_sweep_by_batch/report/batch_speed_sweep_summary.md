# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 1 | dense_default_amp | dense_default_amp | 37.509 | 1.000x |
| 1 | uniform_dense_bf16 | uniform_dense_bf16 | 29.833 | 1.257x |
| 1 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 67.037 | 0.560x |
| 1 | uniform_sparse_bf16 | uniform_sparse_bf16 | 39.726 | 0.944x |
| 1 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 80.683 | 0.465x |
| 1 | ours_best | ours_gate_up_sparse_bf16_64 | 35.037 | 1.071x |
| 2 | dense_default_amp | dense_default_amp | 39.117 | 1.000x |
| 2 | uniform_dense_bf16 | uniform_dense_bf16 | 27.137 | 1.441x |
| 2 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 67.536 | 0.579x |
| 2 | uniform_sparse_bf16 | uniform_sparse_bf16 | 45.741 | 0.855x |
| 2 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 69.225 | 0.565x |
| 2 | ours_best | ours_mlp_sparse_bf16_96 | 37.374 | 1.047x |
| 4 | dense_default_amp | dense_default_amp | 38.648 | 1.000x |
| 4 | uniform_dense_bf16 | uniform_dense_bf16 | 26.926 | 1.435x |
| 4 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 73.335 | 0.527x |
| 4 | uniform_sparse_bf16 | uniform_sparse_bf16 | 49.743 | 0.777x |
| 4 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 72.385 | 0.534x |
| 4 | ours_best | ours_mlp_sparse_bf16_96 | 39.603 | 0.976x |
| 8 | dense_default_amp | dense_default_amp | 41.997 | 1.000x |
| 8 | uniform_dense_bf16 | uniform_dense_bf16 | 29.424 | 1.427x |
| 8 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 75.497 | 0.556x |
| 8 | uniform_sparse_bf16 | uniform_sparse_bf16 | 39.149 | 1.073x |
| 8 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 84.224 | 0.499x |
| 8 | ours_best | ours_mlp_sparse_bf16_96 | 35.980 | 1.167x |
| 16 | dense_default_amp | dense_default_amp | 55.532 | 1.000x |
| 16 | uniform_dense_bf16 | uniform_dense_bf16 | 51.413 | 1.080x |
| 16 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 84.832 | 0.655x |
| 16 | uniform_sparse_bf16 | uniform_sparse_bf16 | 46.725 | 1.188x |
| 16 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 84.204 | 0.659x |
| 16 | ours_best | ours_mlp_sparse_bf16_96 | 42.723 | 1.300x |
| 32 | dense_default_amp | dense_default_amp | 102.905 | 1.000x |
| 32 | uniform_dense_bf16 | uniform_dense_bf16 | 92.393 | 1.114x |
| 32 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 119.333 | 0.862x |
| 32 | uniform_sparse_bf16 | uniform_sparse_bf16 | 71.704 | 1.435x |
| 32 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 117.924 | 0.873x |
| 32 | ours_best | ours_extreme_fastest_microbench | 73.991 | 1.391x |
