# Phase Switch Speed Benchmark

## balanced

| label | generate_s | output tok/s | total tok/s | init_s |
|---|---:|---:|---:|---:|
| dense_nvfp4 | 8.5095 | 481.34 | 962.68 | 9.56 |
| phase_switch | 3.9974 | 1024.67 | 2049.33 | 14.53 |
| w4a16_ours | 3.1687 | 1292.65 | 2585.29 | 8.81 |

best: `w4a16_ours`
phase vs dense_nvfp4 speedup: 2.129x
phase vs w4a16_ours speedup: 0.793x

## decode_heavy

| label | generate_s | output tok/s | total tok/s | init_s |
|---|---:|---:|---:|---:|
| dense_nvfp4 | 4.2894 | 477.46 | 596.83 | 9.01 |
| phase_switch | 2.0994 | 975.52 | 1219.40 | 14.05 |
| w4a16_ours | 1.6000 | 1280.04 | 1600.04 | 9.14 |

best: `w4a16_ours`
phase vs dense_nvfp4 speedup: 2.043x
phase vs w4a16_ours speedup: 0.762x

## prefill_heavy

| label | generate_s | output tok/s | total tok/s | init_s |
|---|---:|---:|---:|---:|
| dense_nvfp4 | 1.5123 | 338.57 | 5755.63 | 12.04 |
| phase_switch | 0.9803 | 522.31 | 8879.35 | 14.92 |
| w4a16_ours | 1.0652 | 480.68 | 8171.57 | 12.04 |

best: `phase_switch`
phase vs dense_nvfp4 speedup: 1.543x
phase vs w4a16_ours speedup: 1.087x
