# Warm E2E Aligned Retest Analysis

## Scope

This run aligns linear-level manual/pred estimates to the current warmed full-model E2E benchmark semantics:

- prefill warmup is performed before timed full-model E2E;
- prefill first materialization is not counted in linear estimates;
- first decode online materialization is still counted.

Only `llama2-7b` under `normal_02` was run in this pass.

## Llama2-7B normal_02 E2E

| method | policy/method | prefill ms | decode x256 ms | e2e ms |
|---|---:|---:|---:|---:|
| pred | pred | 1240.86 | 6041.50 | 7282.37 |
| single | marlin_nvfp4 | 1519.05 | 6198.82 | 7717.87 |
| manual | manual | 1402.17 | 6320.71 | 7722.88 |
| single | dense_bf16 | 1512.00 | 7589.40 | 9101.40 |
| single | sparse_bf16 | 1163.81 | 9170.89 | 10334.71 |
| single | dense_nvfp4 | 1187.08 | 16161.77 | 17348.85 |
| single | sparse_nvfp4 | 1165.92 | 20563.51 | 21729.43 |

## Policy Observations

- `pred` selects `dense_nvfp4->marlin_nvfp4` for all three MLP groups and `marlin_nvfp4->marlin_nvfp4` for attention groups.
- `manual` now selects `dense_nvfp4->marlin_nvfp4` for `mlp.up_proj`, but still selects `marlin_nvfp4->marlin_nvfp4` for `mlp.down_proj` and `mlp.gate_proj`.
- `manual` also selects dense bf16 for `self_attn.o_proj` and `self_attn.q_proj` in this run, which makes the full-model result worse than `pred` and close to single Marlin.

## Main Takeaway

The warm-E2E alignment fix removed prefill first-cost bias from manual scoring, but standalone module measurements still do not fully reproduce full-model E2E ranking. The remaining gap is mainly policy-choice quality from module-level measurements, not the old prefill first-cost accounting.
