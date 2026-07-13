# Actual Prefill-Only Policy Quality Comparison

## Evaluation

- Task: full ARC-Challenge, 0-shot.
- Metric: `acc_norm`, `sample_len=1172`.
- Backend: vLLM/lm-eval results from exported checkpoints.
- Compression note: compressed results use real exported checkpoints built from prepared compressed weights; they are not runtime module replacement results.

## Global Baselines

| method | acc_norm | delta vs dense |
|---|---:|---:|
| `dense_bf16` | 0.4514 | 0.0000 |
| `dense_nvfp4` | 0.4377 | -0.0137 |
| `marlin_nvfp4` | 0.4360 | -0.0154 |
| `sparse_bf16` | 0.3379 | -0.1135 |
| `sparse_nvfp4` | 0.2287 | -0.2227 |

## Speed-Only Hetero Policies

| speed scenario | hetero policy | policy mix | acc_norm | delta vs dense | closest uniform quality |
|---|---|---|---:|---:|---|
| `b8_in2048_out1` | `maxspeed_004_f2600ffcfc` | `dense_nvfp4:64,sparse_bf16:64` | 0.4087 | -0.0427 | below `marlin_nvfp4`/`dense_nvfp4`, above `sparse_bf16` |
| `b1_in512_out1` | `maxspeed_005_4746310a30` | `sparse_bf16:96,sparse_nvfp4:32` | 0.2884 | -0.1630 | below `sparse_bf16`, above `sparse_nvfp4` |

## Interpretation

- `b8_in2048_out1` uses a less aggressive max-speed policy and keeps quality materially better than uniform `sparse_bf16`/`sparse_nvfp4`, but worse than uniform `dense_nvfp4`/`marlin_nvfp4`.
- `b1_in512_out1` is the speed-favorable point, but its policy is aggressive and quality is poor: it is worse than uniform `sparse_bf16`, only better than uniform `sparse_nvfp4`.
- Since accuracy is checkpoint-level, these values apply to the corresponding policy checkpoints rather than depending on the prefill batch/sequence benchmark.
