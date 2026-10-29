# Decomposed Phase Switch Benchmark With BF16

scenario: `phase_balanced_i2048_o80_bs16`

Prefill is approximated by the same prompt batch with `output_len=1`; decode is `output_len=N` total time minus the `output_len=1` run.

| label | prefill_s | decode_s | total_s | prefill tok/s | decode tok/s | total tok/s |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 2.1743 | 2.7822 | 4.9565 | 15070.67 | 454.31 | 6869.34 |
| dense_nvfp4 | 1.3609 | 2.9308 | 4.2917 | 24077.88 | 431.28 | 7933.38 |
| w4a16_ours | 2.2896 | 1.4907 | 3.7803 | 14311.55 | 847.93 | 9006.69 |
| phase_switch_overlap | 1.4978 | 1.6279 | 3.1258 | 21876.84 | 776.45 | 10892.73 |
| phase_switch_sync | 1.4586 | 1.5237 | 2.9823 | 22465.34 | 829.57 | 11416.73 |

## Switch overhead

- non-overlap switch copy: 0.1023s
- exposed boundary wait with overlap: 0.0064s
- hidden by overlap: 0.0959s
- sync total minus overlap total: -0.1435s
