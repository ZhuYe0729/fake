# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 16 | dense_default_amp | dense_default_amp | 55.215 | 1.000x |
| 16 | uniform_dense_bf16 | uniform_dense_bf16 | 51.165 | 1.079x |
| 16 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 82.082 | 0.673x |
| 16 | uniform_sparse_bf16 | uniform_sparse_bf16 | 39.816 | 1.387x |
| 16 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 94.876 | 0.582x |
| 16 | ours_best | ours_mlp_all_attn_sparse_bf16_64 | 41.571 | 1.328x |
