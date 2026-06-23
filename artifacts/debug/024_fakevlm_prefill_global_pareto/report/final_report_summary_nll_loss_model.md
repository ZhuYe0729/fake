# FakeVLM Prefill Pareto Report

Pareto rows are selected mixed policies from `024_fakevlm_prefill_global_pareto` with full FakeClue validation and measured E2E prefill latency.
Uniform rows use FakeClue accuracy from `020_fakevlm_uniform_accuracy`; non-dense uniform speed uses measured prefill latency from `021_fakevlm_linear_hybrid_prefill_speed`.

## Files

- `final_fakevlm_report.csv`
- `pareto_batch_1_speed_vs_fakeclue.png`
- `pareto_batch_1_speed_vs_fakeclue.pdf`
- `pareto_batch_2_speed_vs_fakeclue.png`
- `pareto_batch_2_speed_vs_fakeclue.pdf`
- `pareto_batch_4_speed_vs_fakeclue.png`
- `pareto_batch_4_speed_vs_fakeclue.pdf`
- `pareto_batch_8_speed_vs_fakeclue.png`
- `pareto_batch_8_speed_vs_fakeclue.pdf`
- `pareto_batch_16_speed_vs_fakeclue.png`
- `pareto_batch_16_speed_vs_fakeclue.pdf`

## Batch Summary

| Batch | Dense ms | Fastest point | Fastest speedup | Fastest acc | Best acc point | Best acc |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104.451 | P30 | 1.736 | 0.9528 | P18 | 0.9874 |
| 2 | 183.184 | P30 | 1.836 | 0.7648 | P09 | 0.9872 |
| 4 | 359.210 | P30 | 1.813 | 0.9520 | P05 | 0.9870 |
| 8 | 669.581 | P30 | 1.646 | 0.9530 | P05 | 0.9870 |
| 16 | 1283.530 | P30 | 1.578 | 0.9522 | P22 | 0.9874 |

## Selected Points

| Batch | Point | Speedup | E2E ms | Accuracy | Replaced | Counts |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 1.000 | 104.451 | 0.9864 | 0 | dense_bf16=224 |
| 1 | 5 | 1.007 | 103.721 | 0.9870 | 5 | dense_bf16=219, sparse_bf16=5 |
| 1 | 9 | 1.021 | 102.292 | 0.9866 | 10 | dense_bf16=214, dense_nvfp4=2, sparse_bf16=8 |
| 1 | 13 | 1.045 | 99.925 | 0.9868 | 18 | dense_bf16=206, dense_nvfp4=8, sparse_bf16=10 |
| 1 | 18 | 1.101 | 94.906 | 0.9874 | 38 | dense_bf16=186, dense_nvfp4=25, sparse_bf16=13 |
| 1 | 22 | 1.342 | 77.846 | 0.9870 | 74 | dense_bf16=150, dense_nvfp4=58, sparse_bf16=16 |
| 1 | 26 | 1.548 | 67.470 | 0.9862 | 163 | dense_bf16=61, dense_nvfp4=66, sparse_bf16=97 |
| 1 | 30 | 1.736 | 60.156 | 0.9528 | 224 | sparse_bf16=160, sparse_nvfp4=64 |
| 2 | 0 | 1.000 | 183.184 | 0.9864 | 0 | dense_bf16=224 |
| 2 | 5 | 1.012 | 180.940 | 0.9870 | 5 | dense_bf16=219, sparse_bf16=5 |
| 2 | 9 | 1.027 | 178.446 | 0.9872 | 10 | dense_bf16=214, dense_nvfp4=2, sparse_bf16=7, sparse_nvfp4=1 |
| 2 | 13 | 1.051 | 174.278 | 0.9868 | 21 | dense_bf16=203, dense_nvfp4=9, sparse_bf16=10, sparse_nvfp4=2 |
| 2 | 18 | 1.117 | 163.973 | 0.9872 | 42 | dense_bf16=182, dense_nvfp4=28, sparse_bf16=12, sparse_nvfp4=2 |
| 2 | 22 | 1.297 | 141.229 | 0.9848 | 81 | dense_bf16=143, dense_nvfp4=67, sparse_bf16=12, sparse_nvfp4=2 |
| 2 | 26 | 1.631 | 112.280 | 0.9854 | 193 | dense_bf16=31, dense_nvfp4=156, sparse_bf16=35, sparse_nvfp4=2 |
| 2 | 30 | 1.836 | 99.793 | 0.7648 | 224 | sparse_bf16=32, sparse_nvfp4=192 |
| 4 | 0 | 1.000 | 359.210 | 0.9864 | 0 | dense_bf16=224 |
| 4 | 5 | 1.015 | 353.971 | 0.9870 | 5 | dense_bf16=219, sparse_bf16=5 |
| 4 | 9 | 1.031 | 348.490 | 0.9866 | 10 | dense_bf16=214, dense_nvfp4=2, sparse_bf16=8 |
| 4 | 13 | 1.054 | 340.922 | 0.9864 | 21 | dense_bf16=203, dense_nvfp4=9, sparse_bf16=12 |
| 4 | 18 | 1.124 | 319.498 | 0.9866 | 43 | dense_bf16=181, dense_nvfp4=30, sparse_bf16=13 |
| 4 | 22 | 1.260 | 285.180 | 0.9864 | 81 | dense_bf16=143, dense_nvfp4=67, sparse_bf16=14 |
| 4 | 26 | 1.643 | 218.620 | 0.9852 | 192 | dense_bf16=32, dense_nvfp4=151, sparse_bf16=41 |
| 4 | 30 | 1.813 | 198.144 | 0.9520 | 224 | sparse_bf16=160, sparse_nvfp4=64 |
| 8 | 0 | 1.000 | 669.581 | 0.9864 | 0 | dense_bf16=224 |
| 8 | 5 | 1.015 | 659.438 | 0.9870 | 5 | dense_bf16=219, sparse_bf16=5 |
| 8 | 9 | 1.027 | 652.165 | 0.9866 | 10 | dense_bf16=214, dense_nvfp4=2, sparse_bf16=8 |
| 8 | 13 | 1.045 | 640.927 | 0.9868 | 18 | dense_bf16=206, dense_nvfp4=8, sparse_bf16=10 |
| 8 | 18 | 1.093 | 612.498 | 0.9866 | 38 | dense_bf16=186, dense_nvfp4=25, sparse_bf16=13 |
| 8 | 22 | 1.211 | 552.727 | 0.9860 | 75 | dense_bf16=149, dense_nvfp4=62, sparse_bf16=13 |
| 8 | 26 | 1.455 | 460.188 | 0.9820 | 173 | dense_bf16=51, dense_nvfp4=84, sparse_bf16=89 |
| 8 | 30 | 1.646 | 406.746 | 0.9530 | 224 | sparse_bf16=160, sparse_nvfp4=64 |
| 16 | 0 | 1.000 | 1283.530 | 0.9864 | 0 | dense_bf16=224 |
| 16 | 5 | 1.010 | 1270.662 | 0.9870 | 5 | dense_bf16=219, sparse_bf16=5 |
| 16 | 9 | 1.021 | 1256.870 | 0.9866 | 10 | dense_bf16=214, dense_nvfp4=2, sparse_bf16=8 |
| 16 | 13 | 1.031 | 1245.484 | 0.9868 | 18 | dense_bf16=206, dense_nvfp4=8, sparse_bf16=10 |
| 16 | 18 | 1.086 | 1182.263 | 0.9870 | 38 | dense_bf16=186, dense_nvfp4=25, sparse_bf16=13 |
| 16 | 22 | 1.194 | 1075.406 | 0.9874 | 76 | dense_bf16=148, dense_nvfp4=59, sparse_bf16=17 |
| 16 | 26 | 1.410 | 909.999 | 0.9864 | 165 | dense_bf16=59, dense_nvfp4=69, sparse_bf16=96 |
| 16 | 30 | 1.578 | 813.404 | 0.9522 | 224 | sparse_bf16=160, sparse_nvfp4=64 |

## Uniform Baselines

| Batch | Method | Speedup | E2E ms | Accuracy | Counts |
|---:|---|---:|---:|---:|---|
| 1 | Uniform dense BF16 | 1.000 | 104.451 | 0.9864 | dense_bf16=224 |
| 1 | Uniform dense NVFP4 | 1.471 | 70.988 | 0.9870 | dense_nvfp4=224 |
| 1 | Uniform sparse BF16 | 1.587 | 65.807 | 0.9852 | sparse_bf16=224 |
| 1 | Uniform sparse NVFP4 | 1.495 | 69.889 | 0.7686 | sparse_nvfp4=224 |
| 2 | Uniform dense BF16 | 1.000 | 183.184 | 0.9864 | dense_bf16=224 |
| 2 | Uniform dense NVFP4 | 1.658 | 110.486 | 0.9870 | dense_nvfp4=224 |
| 2 | Uniform sparse BF16 | 1.600 | 114.460 | 0.9852 | sparse_bf16=224 |
| 2 | Uniform sparse NVFP4 | 1.774 | 103.283 | 0.7686 | sparse_nvfp4=224 |
| 4 | Uniform dense BF16 | 1.000 | 359.210 | 0.9864 | dense_bf16=224 |
| 4 | Uniform dense NVFP4 | 1.647 | 218.082 | 0.9870 | dense_nvfp4=224 |
| 4 | Uniform sparse BF16 | 1.666 | 215.560 | 0.9852 | sparse_bf16=224 |
| 4 | Uniform sparse NVFP4 | 1.760 | 204.053 | 0.7686 | sparse_nvfp4=224 |
| 8 | Uniform dense BF16 | 1.000 | 669.581 | 0.9864 | dense_bf16=224 |
| 8 | Uniform dense NVFP4 | 1.377 | 486.160 | 0.9870 | dense_nvfp4=224 |
| 8 | Uniform sparse BF16 | 1.546 | 433.177 | 0.9852 | sparse_bf16=224 |
| 8 | Uniform sparse NVFP4 | 1.459 | 458.955 | 0.7686 | sparse_nvfp4=224 |
| 16 | Uniform dense BF16 | 1.000 | 1283.530 | 0.9864 | dense_bf16=224 |
| 16 | Uniform dense NVFP4 | 1.296 | 990.030 | 0.9870 | dense_nvfp4=224 |
| 16 | Uniform sparse BF16 | 1.486 | 863.468 | 0.9852 | sparse_bf16=224 |
| 16 | Uniform sparse NVFP4 | 1.384 | 927.380 | 0.7686 | sparse_nvfp4=224 |
