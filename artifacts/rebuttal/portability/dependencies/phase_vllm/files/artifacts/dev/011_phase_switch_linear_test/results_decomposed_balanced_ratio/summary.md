# Decomposed Phase Switch Benchmark

scenario: `balanced_ratio_i2048_o128_bs16`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 1.4582 | 5.1092 | 6.5674 | 22471.39 | 397.72 | 5301.36 |
| w4a16_ours | 2.2940 | 2.3639 | 4.6578 | 14284.46 | 859.60 | 7474.70 |
| phase_switch_overlap | 1.5201 | 2.4343 | 3.9544 | 21556.81 | 834.72 | 8804.34 |
| phase_switch_sync | 1.4730 | 2.4041 | 3.8771 | 22245.49 | 845.22 | 8979.83 |

## Switch overhead

- non-overlap switch copy: 0.0995s
- exposed boundary wait with overlap: 0.0065s
- hidden by overlap: 0.0929s
- sync total minus overlap total: -0.0773s
