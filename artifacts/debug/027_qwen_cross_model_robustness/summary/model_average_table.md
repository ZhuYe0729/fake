# Qwen/Llama Cross-Model Average Table

| Model | Average | dense_bf16 | uniform_dense_nvfp4 | uniform_sparse_bf16 | uniform_sparse_nvfp4 | uniform_marlin_weight_only | uniform_dense_nvfp4_prefill_marlin_decode | our_linear_hybrid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.5-0.8B` | `arith_mean` | 1.000x | 0.760x | 0.938x | 0.707x | 0.957x | 0.892x | 1.102x |
| `Qwen3.5-0.8B` | `geomean` | 1.000x | 0.747x | 0.927x | 0.684x | 0.956x | 0.892x | 1.100x |
| `Qwen3.5-2B` | `arith_mean` | 1.000x | 0.790x | 0.963x | 0.721x | 0.926x | 0.945x | 1.036x |
| `Qwen3.5-2B` | `geomean` | 1.000x | 0.772x | 0.947x | 0.693x | 0.925x | 0.942x | 1.027x |
| `Qwen3.5-4B` | `arith_mean` | 1.000x | 0.803x | 1.017x | 0.805x | 0.980x | 1.035x | 1.109x |
| `Qwen3.5-4B` | `geomean` | 1.000x | 0.782x | 1.005x | 0.773x | 0.979x | 1.034x | 1.105x |
| `Qwen3.5-9B` | `arith_mean` | 1.000x | 0.873x | 1.066x | 0.870x | 0.980x | 1.092x | 1.187x |
| `Qwen3.5-9B` | `geomean` | 1.000x | 0.833x | 1.044x | 0.820x | 0.979x | 1.085x | 1.173x |
| `Llama-2-7B` | `arith_mean` | 1.000x | 0.936x | 1.206x | 0.932x | 1.087x | 1.294x | 1.429x |
| `Llama-2-7B` | `geomean` | 1.000x | 0.885x | 1.171x | 0.856x | 1.083x | 1.293x | 1.411x |
| `Llama-3.1-8B` | `arith_mean` | 1.000x | 0.912x | 1.188x | 0.912x | 1.027x | 1.196x | 1.450x |
| `Llama-3.1-8B` | `geomean` | 1.000x | 0.850x | 1.144x | 0.812x | 1.026x | 1.189x | 1.410x |
| `all_models` | `overall_arith_mean` | 1.000x | 0.846x | 1.063x | 0.824x | 0.993x | 1.076x | 1.219x |
| `all_models` | `overall_geomean` | 1.000x | 0.810x | 1.036x | 0.770x | 0.990x | 1.064x | 1.195x |
