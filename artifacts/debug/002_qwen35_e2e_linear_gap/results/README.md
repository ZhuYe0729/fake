# Qwen3.5-9B E2E Linear Gap Trace

- Scenario: `normal_01` -> `{'batch_size': 1, 'input_tokens': 16384, 'output_tokens': 32}`
- `no_hook_*`: normal full-model E2E timing, same model object before hooks.
- `traced_*`: full-model E2E with CUDA event hooks on every compressible linear; this is only for attribution and is expected to be slower.
- `traced_linear_sum_ms`: sum of all measured compressible linear module forwards during the traced run.

## Method Summary

| Method | No-hook E2E | Traced E2E | Traced linear sum | Backend counts |
| --- | ---: | ---: | ---: | --- |
| `sparse_bf16` | 4689.9148 | 4585.4201 | 1479.9217 | `{'sparse_bf16': 248}` |
| `manual` | 3753.8458 | 4288.5404 | 1140.5320 | `{'marlin_nvfp4': 128, 'bf16': 56, 'dense_nvfp4/marlin_nvfp4': 64}` |
| `pred` | 4042.3999 | 4656.9601 | 1099.3386 | `{'marlin_nvfp4': 56, 'dense_nvfp4/marlin_nvfp4': 128, 'bf16': 64}` |

## Largest Group Totals

| Method | Group | Region | Backend | Calls | Total ms | First ms | Max ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `manual` | `mlp.down_proj` | `prefill` | `marlin_nvfp4` | 32 | 221.9070 | 6.9263 | 6.9642 |
| `sparse_bf16` | `mlp.down_proj` | `decode` | `sparse_bf16` | 1024 | 200.7822 | 0.2191 | 0.2319 |
| `pred` | `mlp.down_proj` | `prefill` | `dense_nvfp4` | 32 | 182.9022 | 5.7088 | 5.7334 |
| `sparse_bf16` | `mlp.gate_proj` | `decode` | `sparse_bf16` | 1024 | 118.4706 | 0.1422 | 0.1616 |
| `manual` | `linear_attn.in_proj_qkv` | `prefill` | `marlin_nvfp4` | 24 | 116.0325 | 4.8312 | 4.8558 |
| `sparse_bf16` | `mlp.gate_proj` | `prefill` | `sparse_bf16` | 32 | 114.3511 | 3.5717 | 3.5871 |
| `sparse_bf16` | `mlp.up_proj` | `prefill` | `sparse_bf16` | 32 | 114.1207 | 3.5656 | 3.5789 |
| `sparse_bf16` | `mlp.down_proj` | `prefill` | `sparse_bf16` | 32 | 113.8156 | 3.5656 | 3.5727 |
| `sparse_bf16` | `mlp.up_proj` | `decode` | `sparse_bf16` | 1024 | 111.4497 | 0.1262 | 0.1431 |
| `sparse_bf16` | `linear_attn.out_proj` | `decode` | `sparse_bf16` | 768 | 98.7636 | 0.1726 | 0.4841 |
| `pred` | `mlp.gate_proj` | `prefill` | `dense_nvfp4` | 32 | 94.8392 | 2.9665 | 2.9788 |
| `pred` | `mlp.up_proj` | `prefill` | `dense_nvfp4` | 32 | 94.7746 | 2.9471 | 2.9747 |
| `manual` | `mlp.gate_proj` | `prefill` | `dense_nvfp4` | 32 | 94.7317 | 2.9604 | 2.9829 |
| `manual` | `mlp.up_proj` | `prefill` | `dense_nvfp4` | 32 | 94.7235 | 2.9543 | 2.9809 |
| `sparse_bf16` | `linear_attn.in_proj_qkv` | `decode` | `sparse_bf16` | 768 | 89.6727 | 0.2369 | 0.2369 |
| `sparse_bf16` | `linear_attn.in_proj_z` | `decode` | `sparse_bf16` | 768 | 81.5812 | 0.1420 | 0.1443 |
| `sparse_bf16` | `linear_attn.in_proj_b` | `decode` | `sparse_bf16` | 768 | 81.0942 | 0.1251 | 1.4441 |
| `sparse_bf16` | `linear_attn.in_proj_a` | `decode` | `sparse_bf16` | 768 | 79.0296 | 0.1193 | 0.2331 |
| `pred` | `linear_attn.out_proj` | `decode` | `marlin_nvfp4` | 768 | 66.0023 | 0.1312 | 0.3786 |
| `pred` | `mlp.gate_proj` | `decode` | `marlin_nvfp4` | 1024 | 64.6373 | 0.0744 | 0.2345 |
| `pred` | `linear_attn.in_proj_b` | `decode` | `bf16` | 768 | 61.9582 | 0.1243 | 0.1986 |
| `pred` | `linear_attn.in_proj_qkv` | `prefill` | `dense_nvfp4` | 24 | 59.3544 | 2.4740 | 2.4904 |
| `pred` | `linear_attn.in_proj_z` | `prefill` | `marlin_nvfp4` | 24 | 58.5261 | 2.4436 | 2.4544 |
| `manual` | `mlp.gate_proj` | `decode` | `marlin_nvfp4` | 1024 | 57.8149 | 0.0711 | 0.0742 |
| `manual` | `linear_attn.in_proj_z` | `prefill` | `marlin_nvfp4` | 24 | 57.6791 | 2.4440 | 2.4440 |
| `sparse_bf16` | `linear_attn.in_proj_qkv` | `prefill` | `sparse_bf16` | 24 | 57.6553 | 2.4064 | 2.4115 |
| `pred` | `linear_attn.out_proj` | `prefill` | `marlin_nvfp4` | 24 | 56.7827 | 2.3644 | 2.3685 |
| `manual` | `linear_attn.out_proj` | `prefill` | `marlin_nvfp4` | 24 | 56.7746 | 2.3644 | 2.3685 |
| `pred` | `mlp.up_proj` | `decode` | `marlin_nvfp4` | 1024 | 53.5498 | 0.0563 | 0.1218 |
| `pred` | `mlp.down_proj` | `decode` | `marlin_nvfp4` | 1024 | 53.1638 | 0.0533 | 0.0745 |
| `manual` | `linear_attn.out_proj` | `decode` | `marlin_nvfp4` | 768 | 51.8240 | 0.0953 | 0.1813 |
| `manual` | `mlp.down_proj` | `decode` | `marlin_nvfp4` | 1024 | 51.5201 | 0.0631 | 0.0631 |
| `manual` | `mlp.up_proj` | `decode` | `marlin_nvfp4` | 1024 | 50.9237 | 0.0532 | 0.0949 |
| `manual` | `linear_attn.in_proj_b` | `decode` | `bf16` | 768 | 44.9392 | 0.0923 | 0.3156 |
| `pred` | `linear_attn.in_proj_qkv` | `decode` | `marlin_nvfp4` | 768 | 44.3617 | 0.1218 | 0.1596 |
| `pred` | `linear_attn.in_proj_z` | `decode` | `marlin_nvfp4` | 768 | 39.9464 | 0.0946 | 0.0946 |
| `manual` | `linear_attn.in_proj_qkv` | `decode` | `marlin_nvfp4` | 768 | 39.8791 | 0.1057 | 0.1057 |
| `manual` | `self_attn.q_proj` | `prefill` | `marlin_nvfp4` | 8 | 38.6294 | 4.8271 | 4.8323 |
| `pred` | `linear_attn.in_proj_a` | `decode` | `bf16` | 768 | 35.4062 | 0.0565 | 0.0677 |
| `manual` | `linear_attn.in_proj_z` | `decode` | `marlin_nvfp4` | 768 | 34.9624 | 0.0773 | 0.0773 |
