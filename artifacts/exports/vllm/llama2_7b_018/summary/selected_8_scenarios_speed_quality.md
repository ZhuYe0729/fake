# Llama2-7B selected 8 scenarios vLLM speed and quality

Speedup is median latency speedup versus `dense_bf16` in the same scenario.
Quality column is full ARC-Challenge `acc_norm` when available.

| method | single_long_prefill_short_decode | small_batch_long_prefill | b4_medium_prefill | b4_long_prefill | b8_mixed_long_prefill | b16_mixed | b32_throughput_mixed | b64_high_batch | avg_speedup | arc_c_acc_norm |
|---|---|---|---|---|---|---|---|---|---|---|
| dense_bf16 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 0.4514 |
| dense_nvfp4 | 1.015x | 1.094x | 1.014x | 1.042x | 0.815x | 0.895x | 0.943x | 0.829x | 0.956x | 0.4377 |
| sparse_bf16 | 1.216x | 1.265x | 1.234x | 1.268x | 1.127x | 1.128x | 1.209x | 1.113x | 1.195x | 0.3379 |
| sparse_nvfp4 | 1.052x | 1.125x | 1.033x | 0.983x | 0.754x | 0.950x | 1.073x | 0.951x | 0.990x | 0.2287 |
| marlin_nvfp4 | 1.128x | 1.190x | 1.132x | 1.180x | 1.229x | 1.225x | 1.315x | 1.280x | 1.210x | 0.4360 |
| hetero | 1.203x | 1.283x | 1.245x | 1.275x | 1.211x | 1.231x | 1.318x | 1.252x | 1.252x | 0.4209 |

## Hetero strategy quality by scenario

| scenario | strategy | assignment | ARC-C acc_norm | source |
|---|---|---|---:|---|
| single_long_prefill_short_decode | hetero_strategy_a | qkv/gate_up=dense_nvfp4, o/down=marlin_nvfp4 | 0.4386 | selected8_vllm_lm_eval |
| small_batch_long_prefill | hetero_strategy_a | qkv/gate_up=dense_nvfp4, o/down=marlin_nvfp4 | 0.4386 | selected8_vllm_lm_eval |
| b4_medium_prefill | hetero_strategy_b | qkv/gate_up=dense_nvfp4, o=dense_bf16, down=marlin_nvfp4 | 0.4411 | selected8_vllm_lm_eval |
| b4_long_prefill | hetero_strategy_b | qkv/gate_up=dense_nvfp4, o=dense_bf16, down=marlin_nvfp4 | 0.4411 | selected8_vllm_lm_eval |
| b8_mixed_long_prefill | hetero_strategy_c | qkv/gate_up=sparse_bf16, o=dense_bf16, down=marlin_nvfp4 | 0.4019 | selected8_vllm_lm_eval |
| b16_mixed | hetero_strategy_c | qkv/gate_up=sparse_bf16, o=dense_bf16, down=marlin_nvfp4 | 0.4019 | selected8_vllm_lm_eval |
| b32_throughput_mixed | hetero_strategy_c | qkv/gate_up=sparse_bf16, o=dense_bf16, down=marlin_nvfp4 | 0.4019 | selected8_vllm_lm_eval |
| b64_high_batch | hetero_strategy_c | qkv/gate_up=sparse_bf16, o=dense_bf16, down=marlin_nvfp4 | 0.4019 | selected8_vllm_lm_eval |

## Scenario configs

| scenario | batch | input_len | output_tokens | prefill_M | hetero_strategy |
|---|---:|---:|---:|---:|---|
| single_long_prefill_short_decode | 1 | 8192 | 16 | 8192 | hetero_strategy_a |
| small_batch_long_prefill | 2 | 4096 | 16 | 8192 | hetero_strategy_a |
| b4_medium_prefill | 4 | 2048 | 16 | 8192 | hetero_strategy_b |
| b4_long_prefill | 4 | 4096 | 32 | 16384 | hetero_strategy_b |
| b8_mixed_long_prefill | 8 | 2048 | 64 | 16384 | hetero_strategy_c |
| b16_mixed | 16 | 1024 | 64 | 16384 | hetero_strategy_c |
| b32_throughput_mixed | 32 | 512 | 64 | 16384 | hetero_strategy_c |
| b64_high_batch | 64 | 256 | 128 | 16384 | hetero_strategy_c |

## Notes

- `dense_bf16`, `dense_nvfp4`, `sparse_bf16`, `sparse_nvfp4`, and `marlin_nvfp4` quality are loaded from the existing full ARC-C uniform results.
- Hetero quality is loaded from `quality/selected_8_scenarios/selected8_vllm_quality.csv`; missing values are left as `pending`.
- Speed rows loaded: 48.
