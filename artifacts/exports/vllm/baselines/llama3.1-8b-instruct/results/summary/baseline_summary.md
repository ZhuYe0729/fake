# Llama-3.1-8B-Instruct Baseline Summary

CNN/DM quality uses `cnn_dm_1000`, the fixed 1000-example subset.
IWSLT uses the same Llama-2-chat tokenizer fallback used by the Llama2 baseline because the PMPD Vicuna tokenizer is unavailable locally; treat it as a non-strict PMPD result.
Quality uses the PMPD Claude-style prompt, not the Llama-3.1 native chat template, to preserve direct comparability with the Llama2 baseline.

## Speed

| method | scenario | e2e median ms | TTFT ms | TPOT ms | total tok/s | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense_bf16 | prefill_only | 1143.194 | 1141.421 | 0.000 | 14338.769 | OK |
| dense_bf16 | prefill_decode | 3629.311 | 2284.273 | 17.026 | 9381.394 | OK |
| dense_nvfp4 | prefill_only | 621.731 | 621.628 | 0.000 | 26365.092 | OK |
| dense_nvfp4 | prefill_decode | 3902.383 | 1246.275 | 33.622 | 8724.926 | OK |
| marlin_nvfp4 | prefill_only | 1109.973 | 1111.111 | 0.000 | 14767.923 | OK |
| marlin_nvfp4 | prefill_decode | 3134.703 | 2214.732 | 11.645 | 10861.634 | OK |
| sparse_bf16 | prefill_only | 729.698 | 727.658 | 0.000 | 22464.095 | OK |
| sparse_bf16 | prefill_decode | 3020.965 | 1448.966 | 19.899 | 11270.571 | OK |
| sparse_nvfp4 | prefill_only | 569.010 | 567.982 | 0.000 | 28807.949 | OK |
| sparse_nvfp4 | prefill_decode | 4589.683 | 1142.220 | 43.639 | 7418.377 | OK |

## Quality

| method | dataset | samples | empty | Rouge-L | BERTScore | SacreBLEU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_bf16 | IWSLT | 333 | 0 | 28.146 |  | 10.680 |
| dense_bf16 | cnn_dm_1000 | 1000 | 0 | 19.487 | 83.600 |  |
| dense_bf16 | dsum | 1500 | 0 | 13.784 | 78.187 |  |
| dense_nvfp4 | IWSLT | 333 | 0 | 28.408 |  | 10.312 |
| dense_nvfp4 | cnn_dm_1000 | 1000 | 0 | 16.345 | 84.988 |  |
| dense_nvfp4 | dsum | 1500 | 0 | 8.203 | 81.634 |  |
| marlin_nvfp4 | IWSLT | 333 | 0 | 28.805 |  | 10.403 |
| marlin_nvfp4 | cnn_dm_1000 | 1000 | 0 | 16.108 | 85.061 |  |
| marlin_nvfp4 | dsum | 1500 | 0 | 8.805 | 81.439 |  |
| sparse_bf16 | IWSLT | 333 | 0 | 14.322 |  | 3.276 |
| sparse_bf16 | cnn_dm_1000 | 1000 | 0 | 12.834 | 81.278 |  |
| sparse_bf16 | dsum | 1500 | 0 | 4.049 | 79.128 |  |
| sparse_nvfp4 | IWSLT | 333 | 0 | 0.110 |  | 0.014 |
| sparse_nvfp4 | cnn_dm_1000 | 1000 | 0 | 0.804 | 76.729 |  |
| sparse_nvfp4 | dsum | 1500 | 0 | 0.126 | 78.438 |  |
