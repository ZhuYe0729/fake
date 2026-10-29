# Decomposed Phase Switch Benchmark

scenario: `prefill_heavy_i512_o32_bs16`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| dense_nvfp4 | 0.4279 | 1.0842 | 1.5121 | 19146.07 | 457.47 | 5756.25 |
| w4a16_ours | 0.6419 | 0.4205 | 1.0624 | 12762.11 | 1179.58 | 8192.85 |
| phase_switch_overlap | 0.4599 | 0.5593 | 1.0192 | 17813.36 | 886.87 | 8540.45 |
| phase_switch_sync | 0.5045 | 0.4886 | 0.9931 | 16236.63 | 1015.21 | 8764.40 |

## Switch overhead

- non-overlap switch copy: 0.1010s
- exposed boundary wait with overlap: 0.0051s
- hidden by overlap: 0.0959s
- sync total minus overlap total: -0.0260s
