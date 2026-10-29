# Decomposed Phase Switch Benchmark

scenario: `heavy_i2048_o256_bs8`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 0.7284 | 7.6221 | 8.3505 | 22494.17 | 267.64 | 2207.30 |
| w4a16_ours | 1.1807 | 2.8603 | 4.0410 | 13876.48 | 713.21 | 4561.25 |
| phase_switch_overlap | 0.7874 | 3.6357 | 4.4231 | 20808.13 | 561.10 | 4167.19 |
| phase_switch_sync | 0.8301 | 3.6114 | 4.4416 | 19736.35 | 564.88 | 4149.90 |

## Switch overhead

- non-overlap switch copy: 0.1002s
- exposed boundary wait with overlap: 0.0042s
- hidden by overlap: 0.0960s
- sync total minus overlap total: 0.0184s
