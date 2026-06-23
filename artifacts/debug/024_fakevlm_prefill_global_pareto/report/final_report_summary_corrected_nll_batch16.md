# FakeVLM Prefill Pareto Report

Pareto rows are selected mixed policies from `024_fakevlm_prefill_global_pareto` with full FakeClue validation and measured E2E prefill latency.
Uniform rows use FakeClue accuracy from `020_fakevlm_uniform_accuracy`; non-dense uniform speed uses measured prefill latency from `021_fakevlm_linear_hybrid_prefill_speed`.

## Files

- `final_fakevlm_report.csv`
- `pareto_batch_16_speed_vs_fakeclue.png`
- `pareto_batch_16_speed_vs_fakeclue.pdf`

## Batch Summary

| Batch | Dense ms | Fastest point | Fastest speedup | Fastest acc | Best acc point | Best acc |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1313.568 | P25 | 1.635 | 0.9528 | P08 | 0.9870 |

## Selected Points

| Batch | Point | Speedup | E2E ms | Accuracy | Replaced | Counts |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 0 | 1.000 | 1313.568 | 0.9864 | 0 | dense_bf16=224 |
| 16 | 4 | 1.013 | 1297.144 | 0.9868 | 4 | dense_bf16=220, dense_nvfp4=4 |
| 16 | 8 | 1.031 | 1273.649 | 0.9870 | 11 | dense_bf16=213, dense_nvfp4=11 |
| 16 | 11 | 1.066 | 1232.169 | 0.9864 | 23 | dense_bf16=201, dense_nvfp4=23 |
| 16 | 15 | 1.183 | 1110.211 | 0.9866 | 59 | dense_bf16=165, dense_nvfp4=59 |
| 16 | 18 | 1.372 | 957.341 | 0.9848 | 121 | dense_bf16=103, dense_nvfp4=64, sparse_bf16=57 |
| 16 | 22 | 1.604 | 819.160 | 0.9484 | 224 | dense_nvfp4=34, sparse_bf16=160, sparse_nvfp4=30 |
| 16 | 25 | 1.635 | 803.450 | 0.9528 | 224 | sparse_bf16=160, sparse_nvfp4=64 |

## Uniform Baselines

| Batch | Method | Speedup | E2E ms | Accuracy | Counts |
|---:|---|---:|---:|---:|---|
| 16 | Uniform dense BF16 | 1.000 | 1313.568 | 0.9864 | dense_bf16=224 |
| 16 | Uniform dense NVFP4 | 1.327 | 990.030 | 0.9870 | dense_nvfp4=224 |
| 16 | Uniform sparse BF16 | 1.521 | 863.468 | 0.9852 | sparse_bf16=224 |
| 16 | Uniform sparse NVFP4 | 1.416 | 927.380 | 0.7686 | sparse_nvfp4=224 |
