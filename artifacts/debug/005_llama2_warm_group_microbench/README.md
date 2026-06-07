# Llama2 Warm Group Microbench

## Purpose

This debug run tests whether a warmer, group-level microbenchmark fixes the standalone manual ranking errors seen in `llama2-7b normal_02`.

Compared with the old standalone manual benchmark, this script:

- builds 32 modules for the same linear shape and executes them sequentially;
- avoids per-iteration finite checks inside the timed loop;
- avoids per-candidate `torch.cuda.empty_cache()`;
- keeps first decode materialization in the measured first decode step.

## Result

The warmed group microbenchmark fixes most of the wrong ranking directions:

| group | old manual picked | pred/full-model favored | warm group microbench |
|---|---|---|---|
| `mlp.gate_proj` | `marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` |
| `mlp.up_proj` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` | `dense_nvfp4->marlin_nvfp4` |
| `self_attn.o_proj` | `dense_bf16` | `marlin_nvfp4` | `marlin_nvfp4` |
| `self_attn.q_proj` | `dense_bf16` | `marlin_nvfp4` | `marlin_nvfp4` |
| `mlp.down_proj` | `marlin_nvfp4` | slightly `dense_nvfp4->marlin_nvfp4` in full-model trace | `marlin_nvfp4` |

## Key Numbers

`warm_group_ranking.csv` shows the new ranking. For the old manual-vs-pred contested groups:

- `mlp.gate_proj`: hybrid is 60.62 ms faster than all-Marlin in the warm group microbench.
- `self_attn.o_proj`: Marlin is 15.40 ms faster than dense bf16.
- `self_attn.q_proj`: Marlin is 15.49 ms faster than dense bf16.
- `mlp.down_proj`: all-Marlin is still 25.85 ms faster in this microbench, while the full-model hook trace slightly favored hybrid by 12.89 ms.

## Conclusion

The original manual microbenchmark was too cold and too single-module-oriented. Removing per-iteration output checks and benchmarking a same-shape group fixes the main policy mistakes. The remaining `mlp.down_proj` mismatch is smaller and likely needs direct full-model ablation or a more faithful whole-layer replay.
