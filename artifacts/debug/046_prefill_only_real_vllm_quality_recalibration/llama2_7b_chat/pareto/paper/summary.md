# Llama2-7B-chat prefill-only: solved Pareto validation

All `ours` values use a policy newly solved from the real-vLLM NLL quality model. NLL is measured over the fixed 100 WikiText blocks; speed is the median of five loaded-vLLM prefill runs (batch 8, input 2048). Each reported point has completed all five real-vLLM downstream tasks. Uniform NLL references come from the same fixed-block runtime calibration; each series is normalized to its own measured dense-BF16 reference.

| policy | speed (ms) | speedup | measured ΔNLL | predicted ΔNLL | status |
|---|---:|---:|---:|---:|---|
| point_000 | 1136.5273 | 1.0000 | 0.0000 | 0.0000 | complete |
| point_008 | 1022.0010 | 1.1121 | 0.0083 | 0.0141 | complete |
| point_012 | 844.2790 | 1.3462 | 0.0150 | 0.0477 | complete |
| point_015 | 682.8227 | 1.6645 | 0.2413 | 0.1183 | complete |
| point_018 | 630.9888 | 1.8012 | 0.7322 | 0.2912 | complete |
| point_020 | 593.8361 | 1.9139 | 1.0654 | 0.5312 | complete |
| point_022 | 559.0657 | 2.0329 | 2.4170 | 0.9681 | complete |
| point_023 | 542.9846 | 2.0931 | 4.8092 | 1.2708 | complete |

## Real-vLLM downstream tasks

| policy | WikiText PPL | WinoGrande acc | ARC-Easy norm acc | ARC-Challenge norm acc | MMLU acc |
|---|---:|---:|---:|---:|---:|
| point_000 | 12.2275 | 0.6827 | 0.6839 | 0.4334 | 0.4657 |
| point_008 | 12.4163 | 0.6788 | 0.6801 | 0.4360 | 0.4586 |
| point_012 | 12.4841 | 0.6827 | 0.6738 | 0.4283 | 0.4551 |
| point_015 | 16.7104 | 0.6614 | 0.6692 | 0.4369 | 0.4511 |
| point_018 | 34.3423 | 0.6496 | 0.6460 | 0.4070 | 0.4132 |
| point_020 | 49.3838 | 0.5730 | 0.5816 | 0.3490 | 0.3062 |
| point_022 | 366.7308 | 0.5596 | 0.4996 | 0.2986 | 0.2470 |
| point_023 | 5923.5496 | 0.5107 | 0.3805 | 0.2645 | 0.2292 |
