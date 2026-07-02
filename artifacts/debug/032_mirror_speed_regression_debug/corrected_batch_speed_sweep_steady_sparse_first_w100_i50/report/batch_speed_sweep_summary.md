# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 1 | dense_default_amp | dense_default_amp | 40.483 | 1.000x |
| 1 | uniform_dense_bf16 | uniform_dense_bf16 | 23.807 | 1.700x |
| 1 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 65.945 | 0.614x |
| 1 | uniform_sparse_bf16 | uniform_sparse_bf16 | 39.352 | 1.029x |
| 1 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 66.988 | 0.604x |
| 1 | ours_best | ours_mlp_sparse_bf16_96 | 31.278 | 1.294x |
| 2 | dense_default_amp | dense_default_amp | 39.817 | 1.000x |
| 2 | uniform_dense_bf16 | uniform_dense_bf16 | 25.452 | 1.564x |
| 2 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 71.616 | 0.556x |
| 2 | uniform_sparse_bf16 | uniform_sparse_bf16 | 39.027 | 1.020x |
| 2 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 70.049 | 0.568x |
| 2 | ours_best | ours_mlp_sparse_bf16_96 | 32.425 | 1.228x |
| 4 | dense_default_amp | dense_default_amp | 39.134 | 1.000x |
| 4 | uniform_dense_bf16 | uniform_dense_bf16 | 25.701 | 1.523x |
| 4 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 72.304 | 0.541x |
| 4 | uniform_sparse_bf16 | uniform_sparse_bf16 | 40.946 | 0.956x |
| 4 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 72.362 | 0.541x |
| 4 | ours_best | ours_mlp_sparse_bf16_96 | 33.705 | 1.161x |
| 8 | dense_default_amp | dense_default_amp | 42.186 | 1.000x |
| 8 | uniform_dense_bf16 | uniform_dense_bf16 | 28.328 | 1.489x |
| 8 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 70.309 | 0.600x |
| 8 | uniform_sparse_bf16 | uniform_sparse_bf16 | 36.055 | 1.170x |
| 8 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 73.663 | 0.573x |
| 8 | ours_best | ours_mlp_sparse_bf16_96 | 31.500 | 1.339x |
| 16 | dense_default_amp | dense_default_amp | 56.253 | 1.000x |
| 16 | uniform_dense_bf16 | uniform_dense_bf16 | 51.565 | 1.091x |
| 16 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 80.709 | 0.697x |
| 16 | uniform_sparse_bf16 | uniform_sparse_bf16 | 40.946 | 1.374x |
| 16 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 81.237 | 0.692x |
| 16 | ours_best | ours_mlp_all_attn_sparse_bf16_64 | 41.366 | 1.360x |
| 32 | dense_default_amp | dense_default_amp | 102.917 | 1.000x |
| 32 | uniform_dense_bf16 | uniform_dense_bf16 | 92.516 | 1.112x |
| 32 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 114.339 | 0.900x |
| 32 | uniform_sparse_bf16 | uniform_sparse_bf16 | 71.910 | 1.431x |
| 32 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 112.434 | 0.915x |
| 32 | ours_best | ours_extreme_fastest_microbench | 73.856 | 1.393x |
