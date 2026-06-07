# Qwen3.5-9B Full-Model Linear Trace

## Scenario

- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- Warmup before traced run: `0`
- Timing method: CUDA events in forward pre/post hooks on the real replaced modules during full model prefill/decode.

## Full E2E

| Method | Convert ms | Prefill ms | Decode first ms | Decode x n ms | E2E ms | Backend counts |
|---|---:|---:|---:|---:|---:|---|
| `sparse_bf16` | 15187.0410 | 3157.6890 | 326.7650 | 2085.8447 | 5243.5336 | `` |
| `predictor_hybrid` | 1181.7993 | 2903.5513 | 354.9548 | 1772.1362 | 4675.6875 | `{'dense_nvfp4/marlin_nvfp4': 184, 'bf16': 64}` |

These E2E values include hook synchronization overhead and are for trace interpretation, not final benchmark ranking.

## No-Hook E2E Recheck

The same two methods were also run without linear hooks, using the same `batch_size=1,input_tokens=16384,output_tokens=32` workload and `warmup_iters=0`.

| Method | Prefill ms | Decode x n ms | E2E ms | Replaced | Skipped |
|---|---:|---:|---:|---:|---:|
| `sparse_bf16` | 3061.2698 | 1973.3371 | 5034.6069 | 248 | 0 |
| `predictor_hybrid` | 2736.4338 | 1664.1619 | 4400.5957 | 248 | 0 |

For this strictly matched bs=1 normal scenario, `predictor_hybrid` is faster than `sparse_bf16`; the earlier "sparse_bf16 is faster" conclusion is not reproduced under this workload.

## No-Hook E2E Recheck With Warmup

The same direct E2E comparison was repeated with `warmup_iters=3`, matching the normal benchmark style more closely.

| Method | Prefill ms | Decode x n ms | E2E ms | Replaced | Skipped |
|---|---:|---:|---:|---:|---:|
| `sparse_bf16` | 2643.7686 | 1967.7851 | 4611.5536 | 248 | 0 |
| `predictor_hybrid` | 2511.8738 | 1692.9030 | 4204.7768 | 248 | 0 |

With warmup enabled, `predictor_hybrid` is still faster by 406.78ms E2E in this bs=1 normal scenario.

## Selected Linear Summary

| Method | Layer | Region | Calls | Total ms | Avg ms | First ms | Max ms |
|---|---|---|---:|---:|---:|---:|---:|
| `sparse_bf16` | `language_model.layers.0.linear_attn.in_proj_qkv` | `decode` | 32 | 49.8124 | 1.5566 | 43.5810 | 43.5810 |
| `sparse_bf16` | `language_model.layers.0.linear_attn.in_proj_qkv` | `prefill` | 1 | 52.9662 | 52.9662 | 52.9662 | 52.9662 |
| `sparse_bf16` | `language_model.layers.0.linear_attn.in_proj_z` | `decode` | 32 | 47.4151 | 1.4817 | 43.3351 | 43.3351 |
| `sparse_bf16` | `language_model.layers.0.linear_attn.in_proj_z` | `prefill` | 1 | 44.6099 | 44.6099 | 44.6099 | 44.6099 |
| `sparse_bf16` | `language_model.layers.0.mlp.down_proj` | `decode` | 32 | 49.6719 | 1.5522 | 43.2443 | 43.2443 |
| `sparse_bf16` | `language_model.layers.0.mlp.down_proj` | `prefill` | 1 | 43.1114 | 43.1114 | 43.1114 | 43.1114 |
| `sparse_bf16` | `language_model.layers.0.mlp.gate_proj` | `decode` | 32 | 47.4684 | 1.4834 | 43.3468 | 43.3468 |
| `sparse_bf16` | `language_model.layers.0.mlp.gate_proj` | `prefill` | 1 | 43.8067 | 43.8067 | 43.8067 | 43.8067 |
| `sparse_bf16` | `language_model.layers.3.self_attn.o_proj` | `decode` | 32 | 4.5972 | 0.1437 | 0.2705 | 0.2705 |
| `sparse_bf16` | `language_model.layers.3.self_attn.o_proj` | `prefill` | 1 | 1.2329 | 1.2329 | 1.2329 | 1.2329 |
| `sparse_bf16` | `language_model.layers.3.self_attn.q_proj` | `decode` | 32 | 4.1429 | 0.1295 | 0.1327 | 0.1341 |
| `sparse_bf16` | `language_model.layers.3.self_attn.q_proj` | `prefill` | 1 | 2.3890 | 2.3890 | 2.3890 | 2.3890 |
| `predictor_hybrid` | `language_model.layers.0.linear_attn.in_proj_qkv` | `decode` | 32 | 115.9976 | 3.6249 | 113.1556 | 113.1556 |
| `predictor_hybrid` | `language_model.layers.0.linear_attn.in_proj_qkv` | `prefill` | 1 | 75.2323 | 75.2323 | 75.2323 | 75.2323 |
| `predictor_hybrid` | `language_model.layers.0.linear_attn.in_proj_z` | `decode` | 32 | 3.5876 | 0.1121 | 1.8266 | 1.8266 |
| `predictor_hybrid` | `language_model.layers.0.linear_attn.in_proj_z` | `prefill` | 1 | 3.1930 | 3.1930 | 3.1930 | 3.1930 |
| `predictor_hybrid` | `language_model.layers.0.mlp.down_proj` | `decode` | 32 | 2.6467 | 0.0827 | 1.0014 | 1.0014 |
| `predictor_hybrid` | `language_model.layers.0.mlp.down_proj` | `prefill` | 1 | 5.7324 | 5.7324 | 5.7324 | 5.7324 |
| `predictor_hybrid` | `language_model.layers.0.mlp.gate_proj` | `decode` | 32 | 3.1303 | 0.0978 | 1.1018 | 1.1018 |
| `predictor_hybrid` | `language_model.layers.0.mlp.gate_proj` | `prefill` | 1 | 3.3669 | 3.3669 | 3.3669 | 3.3669 |
| `predictor_hybrid` | `language_model.layers.3.self_attn.o_proj` | `decode` | 32 | 3.4489 | 0.1078 | 1.1361 | 1.1361 |
| `predictor_hybrid` | `language_model.layers.3.self_attn.o_proj` | `prefill` | 1 | 1.9814 | 1.9814 | 1.9814 | 1.9814 |
| `predictor_hybrid` | `language_model.layers.3.self_attn.q_proj` | `decode` | 32 | 2.9475 | 0.0921 | 0.9227 | 0.9227 |
| `predictor_hybrid` | `language_model.layers.3.self_attn.q_proj` | `prefill` | 1 | 2.4802 | 2.4802 | 2.4802 | 2.4802 |

## Files

- `full_model_trace/linear_trace.csv`: per-call hook records.
- `full_model_trace/linear_trace_summary.csv`: per-layer phase summary.
- `full_model_trace/linear_trace.json`: full structured payload.
- `full_model_trace/no_hook_e2e_sparse_vs_predictor.csv`: no-hook E2E recheck.
- `full_model_trace/no_hook_e2e_sparse_vs_predictor_warmup3.csv`: no-hook E2E recheck with warmup.
