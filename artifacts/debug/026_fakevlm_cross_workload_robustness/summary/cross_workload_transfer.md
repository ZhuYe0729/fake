# FakeVLM Cross-Workload Transfer Table

| strategy | method_used | prefill_only_speedup_vs_dense | normal_01_speedup_vs_dense | normal_02_speedup_vs_dense | arith_mean_speedup | geomean_speedup |
|---|---|---:|---:|---:|---:|---:|
| `prefill_only_best_uniform` | `uniform_sparse_bf16` | 1.520x | 1.050x | 0.938x | 1.169x | 1.144x |
| `normal_01_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 1.318x | 1.110x | 1.103x | 1.177x | 1.173x |
| `normal_02_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 1.318x | 1.110x | 1.103x | 1.177x | 1.173x |
| `our_linear_hybrid` | `our_linear_hybrid` | 1.625x | 1.118x | 1.106x | 1.283x | 1.262x |
