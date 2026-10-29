# Decomposed Phase Switch Benchmark

scenario: `max_balanced_i2816_o80_bs16`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 1.7948 | 5.1492 | 6.9440 | 25103.12 | 245.48 | 6672.78 |
| w4a16_ours | 3.0820 | 2.2252 | 5.3072 | 14619.17 | 568.03 | 8730.75 |
| phase_switch_overlap | 2.1462 | 2.4894 | 4.6355 | 20993.80 | 507.76 | 9995.88 |
| phase_switch_sync | 2.2009 | 2.6480 | 4.8488 | 20472.02 | 477.35 | 9556.16 |

## Switch overhead

- non-overlap switch copy: 0.1003s
- exposed boundary wait with overlap: 0.0048s
- hidden by overlap: 0.0955s
- sync total minus overlap total: 0.2133s
