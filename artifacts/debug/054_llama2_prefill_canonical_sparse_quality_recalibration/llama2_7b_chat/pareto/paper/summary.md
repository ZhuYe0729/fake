# Llama2-7B-chat canonical sparse prefill Pareto

`ours` NLL is measured through real vLLM phase-heterogeneous inference on 100 fixed WikiText blocks. Each speed is the median of five loaded-vLLM prefill runs (batch 8, input 2048). Uniform compressed baselines are remeasured through the identical phase runtime; dense BF16 is the shared phase reference.

## Solved mixed policies

| policy | speed (ms) | speedup | real ΔNLL | predicted ΔNLL | residual |
|---|---:|---:|---:|---:|---:|
| point_000 | 1135.0202 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| point_005 | 1109.6299 | 1.0229 | 0.0057 | 0.0043 | 0.0014 |
| point_010 | 994.4366 | 1.1414 | 0.0109 | 0.0186 | -0.0077 |
| point_015 | 693.1970 | 1.6374 | 0.0279 | 0.0830 | -0.0551 |
| point_016 | 666.9983 | 1.7017 | 0.0396 | 0.1121 | -0.0726 |
| point_017 | 652.0115 | 1.7408 | 0.0428 | 0.1512 | -0.1083 |
| point_018 | 629.5052 | 1.8030 | 0.0523 | 0.2038 | -0.1515 |
| point_019 | 608.0095 | 1.8668 | 0.1110 | 0.2763 | -0.1653 |
| point_020 | 593.7462 | 1.9116 | 0.1611 | 0.3729 | -0.2118 |
| point_024 | 545.2521 | 2.0816 | 0.7408 | 0.9893 | -0.2485 |

## Uniform references

| policy | speed (ms) | speedup | real ΔNLL |
|---|---:|---:|---:|
| dense_bf16 | 1135.0202 | 1.0000 | 0.0000 |
| marlin_nvfp4 | 1115.7993 | 1.0172 | 0.0259 |
| sparse_bf16 | 740.3935 | 1.5330 | 0.3457 |
| dense_nvfp4 | 664.4216 | 1.7083 | 0.0421 |
| sparse_nvfp4 | 611.5206 | 1.8561 | 1.0171 |

## Real-vLLM downstream tasks

| family | policy | WikiText PPL | WinoGrande acc | ARC-Easy acc | ARC-Challenge norm acc | MMLU acc |
|---|---|---:|---:|---:|---:|---:|
| uniform | dense_bf16 | 12.2275 | 0.6827 | 0.7281 | 0.4334 | 0.4657 |
| uniform | marlin_nvfp4 | 12.6913 | 0.6882 | 0.7189 | 0.4309 | 0.4566 |
| uniform | sparse_bf16 | 19.8692 | 0.6385 | 0.6385 | 0.3507 | 0.2824 |
| uniform | dense_nvfp4 | 12.9185 | 0.6780 | 0.7113 | 0.4317 | 0.4494 |
| uniform | sparse_nvfp4 | 52.8970 | 0.5043 | 0.3662 | 0.2338 | 0.2309 |
| ours | point_000 | — | — | — | — | — |
| ours | point_005 | — | — | — | — | — |
| ours | point_010 | 12.4267 | 0.6867 | 0.7243 | 0.4420 | 0.4605 |
| ours | point_015 | 12.7584 | 0.6764 | 0.7264 | 0.4317 | 0.4465 |
| ours | point_016 | 12.9270 | 0.6796 | 0.7168 | 0.4147 | 0.4477 |
| ours | point_017 | 12.9499 | 0.6732 | 0.7058 | 0.4369 | 0.4435 |
| ours | point_018 | — | — | — | — | — |
| ours | point_019 | — | — | — | — | — |
| ours | point_020 | 15.3846 | 0.6425 | 0.6890 | 0.4130 | 0.4139 |
| ours | point_024 | 34.0847 | 0.5675 | 0.5316 | 0.2773 | 0.2351 |
