# Final Workload Selection

This table records the proposed final workload configuration. Latency is measured with vLLM median latency. Quality is full ARC-Challenge 0-shot `acc_norm` measured with vLLM + lm-eval for compressed checkpoints, and from the 018 full ARC-C baseline for dense bf16.

## Selected Workloads

| type | batch | input_seq | output_seq | selected_strategy | selected_policy | dense_bf16_ms | selected_ms | speedup_vs_dense | speedup_vs_best_single | selected_acc_norm | dense_bf16_acc_norm | acc_loss_vs_dense |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| prefill-only | 8 | 512 | 1 | max_speed_hetero | maxspeed_004_f2600ffcfc | 271.082 | 121.600 | 2.229x | 1.020x vs sparse_nvfp4 | 0.4087 | 0.4514 | 0.0427 |
| prefill-decoding | 8 | 16384 | 64 | optimized_hetero | policy_003_23b5bafdf0 | 152027.089 | 15766.101 | 9.643x | 2.777x vs sparse_nvfp4 | 0.3660 | 0.4514 | 0.0854 |
| prefill-decoding | 8 | 16384 | 128 | optimized_hetero | policy_002_605d24248e | 295592.915 | 23096.472 | 12.798x | 3.550x vs sparse_nvfp4 | 0.3618 | 0.4514 | 0.0896 |

## Aggregate

| metric | value |
|---|---:|
| average dense_bf16 median latency | 149297.029 ms |
| average selected median latency | 12994.724 ms |
| average speedup vs dense_bf16 | 8.223x |
| average speedup vs best single | 2.449x |
| average selected acc_norm | 0.3788 |
| dense_bf16 acc_norm | 0.4514 |
| average acc_norm loss vs dense_bf16 | 0.0725 |

## Dense BF16 Baseline

| metric | value |
|---|---:|
| ARC-C acc | 0.4292 |
| ARC-C acc_norm | 0.4514 |
| NLL | 2.0395 |
| sample_len | 1172 |

Notes:

- `dense_bf16_ms` is the uncompressed dense bf16 vLLM median latency for the same `(batch, input_seq, output_seq)` scenario.
- `selected_ms` is the measured vLLM median latency for the selected heterogeneous checkpoint.
- `output_seq=1` is used as the prefill-only proxy in the current vLLM benchmark set.
