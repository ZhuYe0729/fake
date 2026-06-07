# Qwen3.5-9B E2E Linear Gap Trace

- Scenario: `normal_01` -> `{'batch_size': 1, 'input_tokens': 16384, 'output_tokens': 32}`
- `no_hook_*`: normal full-model E2E timing, same model object before hooks.
- `traced_*`: full-model E2E with CUDA event hooks on every compressible linear; this is only for attribution and is expected to be slower.
- `traced_linear_sum_ms`: sum of all measured compressible linear module forwards during the traced run.

## Method Summary

| Method | No-hook E2E | Traced E2E | Traced linear sum | Backend counts |
| --- | ---: | ---: | ---: | --- |
| `sparse_bf16` | 3809.7438 | 4173.2884 | 1449.6253 | `{'sparse_bf16': 248}` |

## Largest Group Totals

| Method | Group | Region | Backend | Calls | Total ms | First ms | Max ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `sparse_bf16` | `mlp.down_proj` | `decode` | `sparse_bf16` | 1024 | 196.9109 | 0.2001 | 0.2033 |
| `sparse_bf16` | `mlp.gate_proj` | `prefill` | `sparse_bf16` | 32 | 115.0905 | 3.5789 | 4.0233 |
| `sparse_bf16` | `mlp.down_proj` | `prefill` | `sparse_bf16` | 32 | 114.7915 | 3.5645 | 3.9813 |
| `sparse_bf16` | `mlp.up_proj` | `prefill` | `sparse_bf16` | 32 | 114.6838 | 3.5697 | 3.7990 |
| `sparse_bf16` | `mlp.gate_proj` | `decode` | `sparse_bf16` | 1024 | 114.2432 | 0.1155 | 0.3465 |
| `sparse_bf16` | `mlp.up_proj` | `decode` | `sparse_bf16` | 1024 | 107.7840 | 0.1060 | 0.1236 |
| `sparse_bf16` | `linear_attn.out_proj` | `decode` | `sparse_bf16` | 768 | 94.6629 | 0.1327 | 0.5126 |
| `sparse_bf16` | `linear_attn.in_proj_qkv` | `decode` | `sparse_bf16` | 768 | 86.0129 | 0.1699 | 0.1699 |
| `sparse_bf16` | `linear_attn.in_proj_z` | `decode` | `sparse_bf16` | 768 | 78.7493 | 0.1241 | 0.1241 |
| `sparse_bf16` | `linear_attn.in_proj_b` | `decode` | `sparse_bf16` | 768 | 77.0181 | 0.1111 | 0.1136 |
| `sparse_bf16` | `linear_attn.in_proj_a` | `decode` | `sparse_bf16` | 768 | 76.1743 | 0.1083 | 0.1141 |
| `sparse_bf16` | `linear_attn.in_proj_qkv` | `prefill` | `sparse_bf16` | 24 | 58.8831 | 2.4074 | 3.5594 |
| `sparse_bf16` | `linear_attn.in_proj_z` | `prefill` | `sparse_bf16` | 24 | 30.8235 | 1.3036 | 1.3036 |
| `sparse_bf16` | `self_attn.o_proj` | `decode` | `sparse_bf16` | 256 | 30.4970 | 0.1251 | 0.1641 |
| `sparse_bf16` | `linear_attn.out_proj` | `prefill` | `sparse_bf16` | 24 | 29.9603 | 1.2401 | 1.4992 |
| `sparse_bf16` | `self_attn.q_proj` | `decode` | `sparse_bf16` | 256 | 28.2622 | 0.1094 | 0.1231 |
| `sparse_bf16` | `self_attn.k_proj` | `decode` | `sparse_bf16` | 256 | 27.2819 | 0.1062 | 0.1159 |
| `sparse_bf16` | `self_attn.v_proj` | `decode` | `sparse_bf16` | 256 | 26.6880 | 0.1030 | 0.1148 |
| `sparse_bf16` | `self_attn.q_proj` | `prefill` | `sparse_bf16` | 8 | 19.4335 | 2.3992 | 2.6153 |
| `sparse_bf16` | `self_attn.o_proj` | `prefill` | `sparse_bf16` | 8 | 10.1018 | 1.2401 | 1.4602 |
| `sparse_bf16` | `linear_attn.in_proj_a` | `prefill` | `sparse_bf16` | 24 | 2.9610 | 0.1236 | 0.1378 |
| `sparse_bf16` | `self_attn.k_proj` | `prefill` | `sparse_bf16` | 8 | 2.8856 | 0.3615 | 0.3615 |
| `sparse_bf16` | `linear_attn.in_proj_b` | `prefill` | `sparse_bf16` | 24 | 2.8649 | 0.1304 | 0.1304 |
| `sparse_bf16` | `self_attn.v_proj` | `prefill` | `sparse_bf16` | 8 | 2.8611 | 0.3584 | 0.3584 |
