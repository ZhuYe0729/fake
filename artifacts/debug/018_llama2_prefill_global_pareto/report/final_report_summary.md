# Final Full ARC-Challenge Report

Use these files for presentation; all ARC-Challenge numbers are full-set results with 1172 examples.

## Files

- `pareto_speed_vs_full_arc_c.png`
- `pareto_speed_vs_nll_full_arc_c.png`
- `policy_composition_selected.png`
- `final_full_arc_c_report.csv`

## Main Table

| row | speedup | NLL delta | ARC-C acc_norm | note |
|---|---:|---:|---:|---|
| point_000 | 1.000 | 0.0000 | 0.4514 | Dense BF16 reference. |
| point_013 | 1.059 | 0.0118 | 0.4505 | Low-loss mixed point: small speedup with nearly dense ARC-C. |
| point_019 | 1.191 | 0.0207 | 0.4471 | Intermediate mixed point: shows the smooth low-loss frontier before P020. |
| point_020 | 1.238 | 0.0226 | 0.4462 | Conservative mixed point: better NLL and ARC-C than all-dense NVFP4. |
| point_024 | 1.487 | 0.0974 | 0.4317 | Main mixed point: faster than uniform sparse baselines with much better quality. |
| point_026 | 1.635 | 0.1580 | 0.4096 | Aggressive mixed point: highest recommended speedup before large ARC-C drop. |
| all_marlin_nvfp4 | 0.991 | 0.0547 | 0.4360 | Uniform baseline. |
| all_dense_bf16 | 1.000 | 0.0000 | 0.4514 | Uniform baseline. |
| all_dense_nvfp4 | 1.377 | 0.0820 | 0.4377 | Uniform baseline. |
| all_sparse_bf16 | 1.462 | 0.3503 | 0.3379 | Uniform baseline. |
| all_sparse_nvfp4 | 1.484 | 1.3184 | 0.2287 | Uniform baseline. |

## Suggested Points

- `point_020`: conservative quality point, better NLL and ARC-C than uniform dense NVFP4.
- `point_013` and `point_019`: low-loss supporting points that make the frontier trend easier to see.
- `point_024`: main point, faster than uniform sparse baselines with much better NLL and ARC-C.
- `point_026`: aggressive point, still much better quality than uniform sparse baselines.
