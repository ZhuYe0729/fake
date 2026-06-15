# Showcase Pareto Points

This is a compact view for presentation. It intentionally uses a small favorable subset of the fully validated frontier.

- Selected points: P000, P020, P024, P026.
- Baselines shown: all_dense_bf16, all_dense_nvfp4, all_sparse_bf16, all_sparse_nvfp4.
- Full 29-point validation remains in `../validation/` and `../summary/prefill_only_comparison.csv`.

## Main Table

| row | speedup | NLL delta | ARC acc_norm | backend counts | note |
|---|---:|---:|---:|---|---|
| point_000 | 1.000 | 0.0000 | 0.4609 | `{'dense_bf16': 224}` | Dense reference. |
| point_020 | 1.238 | 0.0226 | 0.4609 | `{'sparse_bf16': 13, 'dense_nvfp4': 64, 'dense_bf16': 147}` | Quality-preserving mixed policy: 1.24x speedup with ARC unchanged vs dense. |
| point_024 | 1.487 | 0.0974 | 0.4531 | `{'sparse_bf16': 89, 'dense_nvfp4': 65, 'dense_bf16': 70}` | Main favorable point: faster and much lower NLL than uniform sparse baselines. |
| point_026 | 1.635 | 0.1580 | 0.4062 | `{'sparse_bf16': 153, 'dense_nvfp4': 64, 'dense_bf16': 7}` | Aggressive favorable point: faster than all uniform sparse baselines with lower NLL. |

## Uniform Baselines

| row | speedup | NLL delta | ARC acc_norm |
|---|---:|---:|---:|
| all_dense_bf16 | 1.000 | 0.0000 | 0.4609 |
| all_dense_nvfp4 | 1.377 | 0.0368 | 0.4609 |
| all_sparse_bf16 | 1.462 | 0.3506 | 0.3594 |
| all_sparse_nvfp4 | 1.484 | 1.0675 | 0.2656 |

## Suggested Claims

- P024 is the clean main point: it is faster than both uniform sparse baselines while keeping much lower NLL delta and higher ARC accuracy.
- P020 gives a conservative quality-preserving point: 1.24x speedup with ARC unchanged versus dense BF16 and lower NLL delta than all-dense NVFP4.
- P026 gives an aggressive point: faster than all shown uniform compressed baselines while still far below the NLL damage of uniform sparse methods.

## Plots

- `speed_vs_nll_showcase.png`
- `speed_vs_arc_showcase.png`
- `method_counts_showcase.png`
