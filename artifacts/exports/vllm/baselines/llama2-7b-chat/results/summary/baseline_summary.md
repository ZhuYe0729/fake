# Llama2-7B-Chat Baseline Summary

CNN/DM quality uses `cnn_dm_1000`, the fixed 1000-example subset.
IWSLT uses the Llama-2-chat tokenizer for length filtering because the PMPD Vicuna tokenizer is unavailable locally; treat it as a non-strict PMPD result.

## Speed

| method | scenario | e2e median ms | TTFT ms | TPOT ms | total tok/s | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense_bf16 | prefill_only | 648.935 | 647.594 | 0.000 | 25259.854 | OK |
| dense_bf16 | prefill_decode | 3140.451 | 1290.921 | 23.412 | 10841.756 | OK |
| dense_nvfp4 | prefill_only | 573.107 | 572.440 | 0.000 | 28602.008 | OK |
| dense_nvfp4 | prefill_decode | 3599.334 | 1162.686 | 30.844 | 9459.529 | OK |
| sparse_bf16 | prefill_only | 482.397 | 480.714 | 0.000 | 33980.309 | OK |
| sparse_bf16 | prefill_decode | 2719.011 | 957.652 | 22.296 | 12522.201 | OK |
| sparse_nvfp4 | prefill_only | 515.776 | 512.589 | 0.000 | 31781.209 | OK |
| sparse_nvfp4 | prefill_decode | 3769.722 | 1031.205 | 34.665 | 9031.965 | OK |
| marlin_nvfp4 | prefill_only | 703.348 | 705.357 | 0.000 | 23305.663 | OK |
| marlin_nvfp4 | prefill_decode | 2744.262 | 1404.979 | 16.953 | 12406.980 | OK |

## Quality

| method | dataset | samples | empty | Rouge-L | BERTScore | SacreBLEU | ARC acc_norm (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_bf16 | IWSLT | 343 | 0 | 46.425 |  | 19.113 |  |
| dense_bf16 | arc_challenge | 1172 |  |  |  |  | 43.174 |
| dense_bf16 | cnn_dm_1000 | 1000 | 0 | 23.682 | 87.410 |  |  |
| dense_bf16 | dsum | 1500 | 0 | 21.688 | 87.669 |  |  |
| dense_nvfp4 | IWSLT | 343 | 0 | 44.828 |  | 16.757 |  |
| dense_nvfp4 | arc_challenge | 1172 |  |  |  |  | 42.833 |
| dense_nvfp4 | cnn_dm_1000 | 1000 | 0 | 24.273 | 87.433 |  |  |
| dense_nvfp4 | dsum | 1500 | 0 | 20.592 | 87.088 |  |  |
| marlin_nvfp4 | IWSLT | 343 | 0 | 46.788 |  | 18.019 |  |
| marlin_nvfp4 | arc_challenge | 1172 |  |  |  |  | 42.833 |
| marlin_nvfp4 | cnn_dm_1000 | 1000 | 0 | 24.505 | 87.476 |  |  |
| marlin_nvfp4 | dsum | 1500 | 0 | 21.241 | 87.403 |  |  |
| sparse_bf16 | IWSLT | 343 | 0 | 27.626 |  | 7.834 |  |
| sparse_bf16 | arc_challenge | 1172 |  |  |  |  | 34.471 |
| sparse_bf16 | cnn_dm_1000 | 1000 | 0 | 15.368 | 84.935 |  |  |
| sparse_bf16 | dsum | 1500 | 189 | 13.452 | 74.311 |  |  |
| sparse_nvfp4 | IWSLT | 343 | 34 | 7.279 |  | 0.889 |  |
| sparse_nvfp4 | arc_challenge | 1172 |  |  |  |  | 23.720 |
| sparse_nvfp4 | cnn_dm_1000 | 1000 | 886 | 1.907 | 9.592 |  |  |
| sparse_nvfp4 | dsum | 1500 | 885 | 5.961 | 34.898 |  |  |
