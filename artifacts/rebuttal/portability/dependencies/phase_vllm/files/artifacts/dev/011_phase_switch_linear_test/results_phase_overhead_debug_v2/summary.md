# Phase Switch Overhead Debug

scenario: `phase_overhead_debug_i2048_o80_bs16`

| label | prefill_s | decode_s | total_s | decode delta | total delta |
|---|---:|---:|---:|---:|---:|
| phase_baseline | 1.3799 | 1.5536 | 2.9335 | 0.0000 | 0.0000 |
| phase_no_trace | 1.4271 | 1.4365 | 2.8636 | -0.1171 | -0.0698 |
| phase_no_decode_gc | 1.4441 | 1.4964 | 2.9405 | -0.0572 | 0.0070 |
| phase_wait_once | 1.4068 | 1.7617 | 3.1685 | 0.2081 | 0.2350 |
| phase_combined | 1.6469 | 1.6052 | 3.2521 | 0.0516 | 0.3186 |
