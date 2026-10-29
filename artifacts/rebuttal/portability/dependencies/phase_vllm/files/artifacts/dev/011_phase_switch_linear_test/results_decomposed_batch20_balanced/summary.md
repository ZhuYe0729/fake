# Decomposed Phase Switch Benchmark

scenario: `batch20_balanced_i2048_o80_bs20`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 1.5909 | 5.0695 | 6.6605 | 25745.84 | 311.67 | 6389.94 |
| w4a16_ours | 2.8068 | 1.7374 | 4.5442 | 14593.10 | 909.40 | 9365.73 |
| phase_switch_overlap | 2.0252 | 2.3201 | 4.3453 | 20224.78 | 681.01 | 9794.44 |
| phase_switch_sync | 1.8537 | 2.5179 | 4.3717 | 22095.93 | 627.50 | 9735.42 |

## Switch overhead

- non-overlap switch copy: 0.0818s
- exposed boundary wait with overlap: 0.0047s
- hidden by overlap: 0.0771s
- sync total minus overlap total: 0.0263s
