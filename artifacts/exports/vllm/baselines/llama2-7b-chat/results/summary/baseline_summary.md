# Llama2-7B-Chat Baseline Summary

CNN/DM quality uses `cnn_dm_1000`, the fixed 1000-example subset.
IWSLT uses the Llama-2-chat tokenizer for length filtering because the PMPD Vicuna tokenizer is unavailable locally; treat it as a non-strict PMPD result.

## Speed

| method | scenario | e2e median ms | TTFT ms | TPOT ms | total tok/s | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense_bf16 | prefill_only | 1079.536 | 1076.834 | 0.000 | 15184.303 | OK |
| dense_bf16 | prefill_decode | 4868.068 | 2151.099 | 34.392 | 6994.151 | OK |
| dense_nvfp4 | prefill_only | 578.163 | 580.941 | 0.000 | 28351.847 | OK |
| dense_nvfp4 | prefill_decode | 4293.725 | 1157.180 | 39.703 | 7929.712 | OK |
| marlin_nvfp4 | prefill_only | 1034.992 | 1033.006 | 0.000 | 15837.807 | OK |
| marlin_nvfp4 | prefill_decode | 3495.347 | 2062.112 | 18.142 | 9740.951 | OK |
| sparse_bf16 | prefill_only | 660.948 | 656.414 | 0.000 | 24800.727 | OK |
| sparse_bf16 | prefill_decode | 3424.870 | 1299.185 | 26.907 | 9941.400 | OK |
| sparse_nvfp4 | prefill_only | 520.632 | 520.293 | 0.000 | 31484.826 | OK |
| sparse_nvfp4 | prefill_decode | 4723.443 | 1051.052 | 46.486 | 7208.302 | OK |

## Quality

| method | dataset | samples | empty | Rouge-L | BERTScore | SacreBLEU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_bf16 | IWSLT | 333 | 0 | 46.701 |  | 19.296 |
| dense_bf16 | cnn_dm_1000 | 1000 | 0 | 23.671 | 87.185 |  |
| dense_bf16 | dsum | 1500 | 0 | 21.688 | 87.176 |  |
| dense_nvfp4 | IWSLT | 333 | 0 | 44.969 |  | 16.835 |
| dense_nvfp4 | cnn_dm_1000 | 1000 | 0 | 24.273 | 87.204 |  |
| dense_nvfp4 | dsum | 1500 | 0 | 20.592 | 86.691 |  |
| marlin_nvfp4 | IWSLT | 333 | 0 | 46.772 |  | 18.200 |
| marlin_nvfp4 | cnn_dm_1000 | 1000 | 0 | 24.579 | 87.251 |  |
| marlin_nvfp4 | dsum | 1500 | 0 | 21.368 | 87.053 |  |
| sparse_bf16 | IWSLT | 333 | 0 | 15.798 |  | 3.900 |
| sparse_bf16 | cnn_dm_1000 | 1000 | 23 | 15.350 | 82.131 |  |
| sparse_bf16 | dsum | 1500 | 156 | 13.539 | 75.392 |  |
| sparse_nvfp4 | IWSLT | 333 | 288 | 1.133 |  | 0.238 |
| sparse_nvfp4 | cnn_dm_1000 | 1000 | 886 | 2.047 | 9.626 |  |
| sparse_nvfp4 | dsum | 1500 | 399 | 9.388 | 61.906 |  |
