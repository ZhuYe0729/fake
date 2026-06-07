# Qwen3.5-9B Multi-Linear Kernel Breakdown

## Scenario

- Model: `Qwen3.5-9B`
- Workload: `batch_size=1, input_tokens=16384, output_tokens=32`
- M: `prefill=16384, decode=1`
- GPU: `NVIDIA GeForce RTX 5090`

## Runtime-Only Steady Summary

| Layer | Shape | sparse_bf16 ms | explicit dense_nvfp4+marlin ms | lazy wrapper ms | sparse - explicit ms |
|---|---:|---:|---:|---:|---:|
| `language_model.layers.0.linear_attn.in_proj_qkv` | `N=8192,K=4096` | 6.628342 | 4.000650 | 4.152557 | 2.627693 |
| `language_model.layers.0.linear_attn.in_proj_z` | `N=4096,K=4096` | 5.316989 | 3.392138 | 3.638851 | 1.924851 |
| `language_model.layers.0.mlp.gate_proj` | `N=12288,K=4096` | 7.663965 | 4.379888 | 4.600701 | 3.284077 |
| `language_model.layers.0.mlp.down_proj` | `N=4096,K=12288` | 10.582336 | 7.109894 | 7.397840 | 3.472441 |
| `language_model.layers.3.self_attn.q_proj` | `N=8192,K=4096` | 6.447539 | 3.877779 | 4.107990 | 2.569760 |
| `language_model.layers.3.self_attn.o_proj` | `N=4096,K=4096` | 5.399866 | 3.402323 | 3.648413 | 1.997542 |

## Prefill/Decode Detail

| Layer | Path | Prefill steady ms | Decode steady ms | Decode x32 ms | Runtime steady ms | Build+conversion ms |
|---|---|---:|---:|---:|---:|---:|
| `language_model.layers.0.linear_attn.in_proj_qkv` | `sparse_bf16` | 2.503875 | 0.128890 | 4.124467 | 6.628342 | 941.216203 |
| `language_model.layers.0.linear_attn.in_proj_qkv` | `dense_nvfp4_prefill_marlin_decode_explicit` | 2.492605 | 0.047126 | 1.508045 | 4.000650 | 398.957459 |
| `language_model.layers.0.linear_attn.in_proj_qkv` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.478624 | 0.052310 | 1.673933 | 4.152557 | 2.160860 |
| `language_model.layers.0.linear_attn.in_proj_z` | `sparse_bf16` | 1.294922 | 0.125690 | 4.022067 | 5.316989 | 88.719905 |
| `language_model.layers.0.linear_attn.in_proj_z` | `dense_nvfp4_prefill_marlin_decode_explicit` | 1.997859 | 0.043571 | 1.394278 | 3.392138 | 2.499780 |
| `language_model.layers.0.linear_attn.in_proj_z` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 1.999427 | 0.051232 | 1.639424 | 3.638851 | 0.973126 |
| `language_model.layers.0.mlp.gate_proj` | `sparse_bf16` | 3.630326 | 0.126051 | 4.033638 | 7.663965 | 94.342838 |
| `language_model.layers.0.mlp.gate_proj` | `dense_nvfp4_prefill_marlin_decode_explicit` | 2.949155 | 0.044710 | 1.430733 | 4.379888 | 4.286833 |
| `language_model.layers.0.mlp.gate_proj` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.954416 | 0.051446 | 1.646285 | 4.600701 | 3.075346 |
| `language_model.layers.0.mlp.down_proj` | `sparse_bf16` | 3.618214 | 0.217629 | 6.964122 | 10.582336 | 93.394190 |
| `language_model.layers.0.mlp.down_proj` | `dense_nvfp4_prefill_marlin_decode_explicit` | 5.729645 | 0.043133 | 1.380250 | 7.109894 | 4.426197 |
| `language_model.layers.0.mlp.down_proj` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 5.740291 | 0.051798 | 1.657549 | 7.397840 | 3.119632 |
| `language_model.layers.3.self_attn.q_proj` | `sparse_bf16` | 2.466842 | 0.124397 | 3.980698 | 6.447539 | 88.277384 |
| `language_model.layers.3.self_attn.q_proj` | `dense_nvfp4_prefill_marlin_decode_explicit` | 2.472032 | 0.043930 | 1.405747 | 3.877779 | 3.191697 |
| `language_model.layers.3.self_attn.q_proj` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.479318 | 0.050896 | 1.628672 | 4.107990 | 2.076549 |
| `language_model.layers.3.self_attn.o_proj` | `sparse_bf16` | 1.296390 | 0.128234 | 4.103475 | 5.399866 | 84.412912 |
| `language_model.layers.3.self_attn.o_proj` | `dense_nvfp4_prefill_marlin_decode_explicit` | 1.995757 | 0.043955 | 1.406566 | 3.402323 | 2.473106 |
| `language_model.layers.3.self_attn.o_proj` | `dense_nvfp4_prefill_marlin_decode_lazy_wrapper` | 2.000797 | 0.051488 | 1.647616 | 3.648413 | 0.944726 |

## Notes

- This debug run targets representative Qwen3.5-9B layers whose predictor policy used `dense_nvfp4/marlin_nvfp4`.
- `runtime steady` excludes offline build/conversion and uses warmed forward latency.
- Per-layer subdirectories contain full JSON/CSV breakdowns.

## Files

- `results/aggregate_breakdown.csv`: flat multi-layer table.
- `results/aggregate_breakdown.json`: structured multi-layer data.
- `results/<layer>/breakdown.csv`: per-layer detail.
