# Decomposed Phase Switch Benchmark

scenario: `balanced_i1024_o256_bs8`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 0.4093 | 7.0489 | 7.4583 | 20012.38 | 289.41 | 1372.97 |
| w4a16_ours | 0.6245 | 2.4776 | 3.1020 | 13118.41 | 823.38 | 3301.04 |
| phase_switch_overlap | 0.4178 | 3.3512 | 3.7689 | 19608.20 | 608.75 | 2716.95 |
| phase_switch_sync | 0.4821 | 3.2344 | 3.7165 | 16992.84 | 630.73 | 2755.31 |

## Switch overhead

- non-overlap switch copy: 0.0822s
- exposed boundary wait with overlap: 0.0025s
- hidden by overlap: 0.0798s
- sync total minus overlap total: -0.0525s
