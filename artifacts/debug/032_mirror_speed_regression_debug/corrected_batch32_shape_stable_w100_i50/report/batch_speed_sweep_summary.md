# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 32 | dense_default_amp | dense_default_amp | 102.810 | 1.000x |
| 32 | uniform_dense_bf16 | uniform_dense_bf16 | 92.397 | 1.113x |
| 32 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 114.590 | 0.897x |
| 32 | uniform_sparse_bf16 | uniform_sparse_bf16 | 72.047 | 1.427x |
| 32 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 108.004 | 0.952x |
| 32 | ours_best | ours_extreme_fastest_shape_stable | 73.565 | 1.398x |
