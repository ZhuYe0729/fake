# MIRROR Batch Speed Sweep

| batch | method | source policy | mean ms | speedup vs uncompressed AMP |
|---:|---|---|---:|---:|
| 1 | dense_default_amp | dense_default_amp | 34.950 | 1.000x |
| 1 | uniform_dense_bf16 | uniform_dense_bf16 | 28.720 | 1.217x |
| 1 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 77.375 | 0.452x |
| 1 | uniform_sparse_bf16 | uniform_sparse_bf16 | 44.186 | 0.791x |
| 1 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 66.641 | 0.524x |
| 1 | ours_best | ours_gate_up_sparse_bf16_64 | 33.393 | 1.047x |
| 2 | dense_default_amp | dense_default_amp | 40.505 | 1.000x |
| 2 | uniform_dense_bf16 | uniform_dense_bf16 | 25.658 | 1.579x |
| 2 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 77.892 | 0.520x |
| 2 | uniform_sparse_bf16 | uniform_sparse_bf16 | 41.395 | 0.979x |
| 2 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 69.232 | 0.585x |
| 2 | ours_best | ours_gate_up_sparse_bf16_64 | 31.381 | 1.291x |
| 4 | dense_default_amp | dense_default_amp | 39.529 | 1.000x |
| 4 | uniform_dense_bf16 | uniform_dense_bf16 | 24.929 | 1.586x |
| 4 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 79.481 | 0.497x |
| 4 | uniform_sparse_bf16 | uniform_sparse_bf16 | 40.773 | 0.969x |
| 4 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 70.375 | 0.562x |
| 4 | ours_best | ours_gate_up_sparse_bf16_64 | 31.729 | 1.246x |
| 8 | dense_default_amp | dense_default_amp | 43.266 | 1.000x |
| 8 | uniform_dense_bf16 | uniform_dense_bf16 | 28.523 | 1.517x |
| 8 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 72.397 | 0.598x |
| 8 | uniform_sparse_bf16 | uniform_sparse_bf16 | 36.726 | 1.178x |
| 8 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 74.944 | 0.577x |
| 8 | ours_best | ours_gate_up_sparse_bf16_64 | 26.106 | 1.657x |
| 16 | dense_default_amp | dense_default_amp | 55.117 | 1.000x |
| 16 | uniform_dense_bf16 | uniform_dense_bf16 | 51.513 | 1.070x |
| 16 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 83.486 | 0.660x |
| 16 | uniform_sparse_bf16 | uniform_sparse_bf16 | 41.434 | 1.330x |
| 16 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 78.398 | 0.703x |
| 16 | ours_best | ours_extreme_fastest_microbench | 41.022 | 1.344x |
| 32 | dense_default_amp | dense_default_amp | 102.892 | 1.000x |
| 32 | uniform_dense_bf16 | uniform_dense_bf16 | 92.437 | 1.113x |
| 32 | uniform_dense_nvfp4 | uniform_dense_nvfp4 | 115.044 | 0.894x |
| 32 | uniform_sparse_bf16 | uniform_sparse_bf16 | 71.867 | 1.432x |
| 32 | uniform_sparse_nvfp4 | uniform_sparse_nvfp4 | 120.592 | 0.853x |
| 32 | ours_best | ours_extreme_fastest_microbench | 73.916 | 1.392x |
