# Llama2 Normal-02 Pareto Smoke Summary

Scenario: batch_size=1, input_tokens=16384, output_tokens=256

## E2E Validation Results

- Points validated: 3
- replaced_linear_count == 224: True
- skipped_linear_count == 0: True

| Point | Pred Total (ms) | E2E Mean (ms) | E2E Median (ms) | Speedup vs Dense | Backends |
|-------|----------------|---------------|-----------------|------------------|----------|
| 0 | 4176.5 | 8950.0 | 8925.4 | 1.0000 | {'bf16': 224} |
| 7 | 3190.3 | 8078.5 | 8073.1 | 1.1079 | {'bf16': 153, 'dense_nvfp4/marlin_nvfp4': 71} |
| 9 | 2829.1 | 7246.4 | 7253.1 | 1.2351 | {'marlin_nvfp4': 128, 'dense_nvfp4/marlin_nvfp4': 96} |

## Failed Smoke Points

| Point | Pred Total (ms) | Backends | Reason |
|-------|-----------------|----------|--------|
| 4 | 3957.5 | {'bf16': 208, 'dense_nvfp4/marlin_nvfp4': 16} | benchmark_failed: CUDA out of memory. Tried to allocate 344.00 MiB. GPU 0 has a total capacity of 31.37 GiB of which 45.75 MiB is free. Including non-PyTorch memory, this proces... |
| 5 | 3800.0 | {'bf16': 197, 'dense_nvfp4/marlin_nvfp4': 27} | benchmark_failed: CUDA out of memory. Tried to allocate 344.00 MiB. GPU 0 has a total capacity of 31.37 GiB of which 309.75 MiB is free. Including non-PyTorch memory, this proce... |
| 6 | 3562.1 | {'bf16': 175, 'marlin_nvfp4': 5, 'dense_nvfp4/marlin_nvfp4': 44} | benchmark_failed: CUDA out of memory. Tried to allocate 344.00 MiB. GPU 0 has a total capacity of 31.37 GiB of which 65.75 MiB is free. Including non-PyTorch memory, this proces... |
| 8 | 2857.2 | {'marlin_nvfp4': 72, 'bf16': 56, 'dense_nvfp4/marlin_nvfp4': 96} | benchmark_failed: CUDA out of memory. Tried to allocate 688.00 MiB. GPU 0 has a total capacity of 31.37 GiB of which 669.75 MiB is free. Including non-PyTorch memory, this proce... |

## Quality Validation Results

| Point | Quality Cost | NLL | NLL Delta | ARC Acc | ARC Acc Norm |
|-------|--------------|-----|-----------|---------|--------------|
| 0 | 0.0000 | 2.039499 | 0.000000 | 0.40625 | 0.4609375 |
| 4 | 0.6481 | 2.064522 | 0.025023 | 0.40625 | 0.46875 |
| 5 | 1.3055 | 2.064560 | 0.025061 | 0.421875 | 0.4609375 |
| 6 | 2.5973 | 2.065575 | 0.026076 | 0.40625 | 0.4609375 |
| 7 | 5.2974 | 2.068435 | 0.028936 | 0.4296875 | 0.4609375 |
| 8 | 10.4523 | 2.072573 | 0.033074 | 0.40625 | 0.4453125 |
| 9 | 16.5301 | 2.076295 | 0.036796 | 0.40625 | 0.4609375 |

## Ranking Check

- Predicted ranking matches E2E ranking: True

## Comparison to Baselines

| Label | E2E Mean (ms) | Speedup vs Dense |
|-------|---------------|------------------|
| point_000 | 8949.952288309732 | 1.0 |
| point_004 |  |  |
| point_005 |  |  |
| point_006 |  |  |
| point_007 | 8078.5079917907715 | 1.1078719359322917 |
| point_008 |  |  |
| point_009 | 7246.382509867351 | 1.2350924445573557 |
| all_dense_bf16 | 9101 | 1.0 |
| all_dense_nvfp4 | 17349 | 0.5245835494841201 |
| all_dense_nvfp4_prefill_marlin_decode | 7762 | 1.172507085802628 |
| all_marlin_nvfp4 | 7718 | 1.1791915003887017 |
| all_sparse_bf16 | 10335 | 0.8805999032414127 |
| all_sparse_nvfp4 | 21729 | 0.4188411799898753 |
| pred_policy | 7282 | 1.2497940126338918 |
| oracle_policy | 7427 | 1.2253938333108927 |

## Acceptability Check

- replaced_linear_count OK: True
- skipped_linear_count OK: True
- Ranking matches: True
