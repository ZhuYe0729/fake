# Promising Scenario Optimized Policy Retest

本表复用 broad-grid 已测 single 方法和原 `hetero` 结果，只新增测试 P024 精度预算下重新求解的 `optimized_hetero` vLLM checkpoint。

## Aggregate

- Mean `optimized_hetero` speedup vs dense bf16: 3.776x.
- Mean `optimized_hetero` speedup vs best single: 1.151x (3/16 scenarios > 1).
- Mean `optimized_hetero` speedup vs original hetero: 1.426x.
- Scenarios faster than best single: b8_in16384_out128 (3.550x), b8_in16384_out64 (2.777x), b64_in512_out16 (1.002x).

## Summary Table

| scenario | best_single | original_hetero_ms | optimized_hetero_ms | opt_vs_dense | opt_vs_best_single | opt_vs_original_hetero | policy | quality_cost | method_counts |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| b2_in16384_out128 | marlin_nvfp4:5392.081 | 5393.556 | 6339.181 | 7.255 | 0.851 | 0.851 | policy_000_4f5ae62f3f | 0.216190828861 | dense=64, dnvfp4=64, sbf16=0, snvfp4=0 |
| b2_in16384_out64 | sparse_bf16:3802.114 | 3786.937 | 4253.670 | 5.653 | 0.894 | 0.890 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b4_in16384_out128 | sparse_nvfp4:31531.113 | 38455.919 | 34973.885 | 3.695 | 0.902 | 1.100 | policy_000_4f5ae62f3f | 0.216190828861 | dense=64, dnvfp4=64, sbf16=0, snvfp4=0 |
| b4_in16384_out64 | sparse_nvfp4:17248.284 | 21192.516 | 18219.566 | 3.664 | 0.947 | 1.163 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b8_in16384_out128 | sparse_nvfp4:81987.028 | 104655.532 | 23096.472 | 12.798 | 3.550 | 4.531 | policy_002_605d24248e | 0.271049145648 | dense=54, dnvfp4=1, sbf16=73, snvfp4=0 |
| b8_in16384_out64 | sparse_nvfp4:43778.766 | 55962.708 | 15766.101 | 9.643 | 2.777 | 3.550 | policy_003_23b5bafdf0 | 0.270692073268 | dense=53, dnvfp4=1, sbf16=74, snvfp4=0 |
| b4_in1024_out1 | sparse_nvfp4:115.757 | 153.884 | 141.429 | 2.021 | 0.818 | 1.088 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b2_in1024_out1 | sparse_nvfp4:64.257 | 81.105 | 72.836 | 2.166 | 0.882 | 1.114 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b4_in512_out1 | sparse_nvfp4:67.543 | 84.882 | 79.721 | 1.937 | 0.847 | 1.065 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b2_in4096_out1 | sparse_nvfp4:261.682 | 325.218 | 315.254 | 1.875 | 0.830 | 1.032 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b1_in4096_out1 | sparse_nvfp4:131.619 | 164.959 | 159.483 | 1.858 | 0.825 | 1.034 | policy_001_2ac520a243 | 0.266931300353 | dense=32, dnvfp4=96, sbf16=0, snvfp4=0 |
| b8_in512_out1 | sparse_nvfp4:123.973 | 192.389 | 140.165 | 1.934 | 0.884 | 1.373 | policy_004_4831f76351 | 0.267393081971 | dense=30, dnvfp4=83, sbf16=15, snvfp4=0 |
| b128_in256_out64 | sparse_bf16:3022.167 | 3122.686 | 3720.446 | 1.443 | 0.812 | 0.839 | policy_005_023ac1246d | 0.277551275605 | dense=62, dnvfp4=12, sbf16=54, snvfp4=0 |
| b32_in4096_out16 | sparse_nvfp4:6031.007 | 7657.591 | 7703.716 | 1.357 | 0.783 | 0.994 | policy_006_4db717c6ec | 0.270673371768 | dense=53, dnvfp4=2, sbf16=73, snvfp4=0 |
| b256_in512_out16 | sparse_nvfp4:5805.582 | 7244.743 | 7214.459 | 1.394 | 0.805 | 1.004 | policy_006_4db717c6ec | 0.270673371768 | dense=53, dnvfp4=2, sbf16=73, snvfp4=0 |
| b64_in512_out16 | sparse_nvfp4:1563.301 | 1842.454 | 1559.841 | 1.717 | 1.002 | 1.181 | policy_007_f74a83b67c | 0.267033012553 | dense=34, dnvfp4=64, sbf16=30, snvfp4=0 |

## Notes

- `optimized_hetero` policies use the 018 P024 quality budget and exclude Marlin from the optimizer because no trusted Marlin quality proxy exists in that run.
- Single-method latency and original hetero latency are copied from `broad_grid_vllm/results/summary_long.csv`.
- `opt_vs_best_single > 1` means the new quality-constrained mixed policy is faster than the best already-tested single method for that scenario.
