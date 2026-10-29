# Decomposed Phase Switch Benchmark

scenario: `scale_balanced_i2560_o100_bs16`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 1.6569 | 6.2235 | 7.8804 | 24720.92 | 254.52 | 5400.76 |
| w4a16_ours | 2.8414 | 2.2426 | 5.0840 | 14415.62 | 706.32 | 8371.43 |
| phase_switch_overlap | 1.9145 | 2.9281 | 4.8426 | 21394.71 | 540.96 | 8788.66 |
| phase_switch_sync | 1.9266 | 2.9568 | 4.8834 | 21260.69 | 535.71 | 8715.22 |

## Switch overhead

- non-overlap switch copy: 0.0816s
- exposed boundary wait with overlap: 0.0048s
- hidden by overlap: 0.0767s
- sync total minus overlap total: 0.0408s
