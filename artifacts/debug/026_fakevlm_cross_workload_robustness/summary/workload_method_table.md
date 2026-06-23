# FakeVLM Cross-Workload Method Table

Cells show `speedup_vs_dense_bf16 (e2e_ms)`.

| Scenario | dense_bf16 | uniform_dense_nvfp4 | uniform_sparse_bf16 | uniform_sparse_nvfp4 | uniform_marlin_weight_only | uniform_dense_nvfp4_prefill_marlin_decode | our_linear_hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prefill_only` | 1.000x (1294.3 ms) | 1.315x (984.5 ms) | 1.520x (851.8 ms) | 1.396x (926.9 ms) | 0.990x (1307.0 ms) | 1.318x (982.4 ms) | 1.625x (796.7 ms) |
| `normal_01` | 1.000x (4919.7 ms) | 0.915x (5376.7 ms) | 1.050x (4686.4 ms) | 0.890x (5530.8 ms) | 1.040x (4732.2 ms) | 1.110x (4432.6 ms) | 1.118x (4400.8 ms) |
| `normal_02` | 1.000x (18125.9 ms) | 0.778x (23294.5 ms) | 0.938x (19327.8 ms) | 0.728x (24893.5 ms) | 1.078x (16818.7 ms) | 1.103x (16432.2 ms) | 1.106x (16386.8 ms) |
| `arith_mean` | 1.000x | 1.003x | 1.169x | 1.005x | 1.036x | 1.177x | 1.283x |
| `geomean` | 1.000x | 0.978x | 1.144x | 0.967x | 1.035x | 1.173x | 1.262x |
