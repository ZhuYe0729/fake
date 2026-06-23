# Qwen/Llama Cross-Model Transfer Table

| strategy | method_used | 0.8B_geomean_speedup | 2B_geomean_speedup | 4B_geomean_speedup | 9B_geomean_speedup | llama2-7b_geomean_speedup | llama31-8b_geomean_speedup | arith_mean_speedup | geomean_speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.8B_best_uniform` | `uniform_marlin_weight_only` | 0.956x | 0.925x | 0.979x | 0.979x | 1.083x | 1.026x | 0.992x | 0.990x |
| `2B_best_uniform` | `uniform_sparse_bf16` | 0.927x | 0.947x | 1.005x | 1.044x | 1.171x | 1.144x | 1.040x | 1.036x |
| `4B_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 0.892x | 0.942x | 1.034x | 1.085x | 1.293x | 1.189x | 1.072x | 1.064x |
| `9B_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 0.892x | 0.942x | 1.034x | 1.085x | 1.293x | 1.189x | 1.072x | 1.064x |
| `llama2-7b_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 0.892x | 0.942x | 1.034x | 1.085x | 1.293x | 1.189x | 1.072x | 1.064x |
| `llama31-8b_best_uniform` | `uniform_dense_nvfp4_prefill_marlin_decode` | 0.892x | 0.942x | 1.034x | 1.085x | 1.293x | 1.189x | 1.072x | 1.064x |
| `our_linear_hybrid` | `our_linear_hybrid` | 1.100x | 1.027x | 1.105x | 1.173x | 1.411x | 1.410x | 1.204x | 1.195x |
