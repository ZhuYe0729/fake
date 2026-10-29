# 开销与摊销边界

阶段耗时分开报告；并行进程耗时之和不等于实际墙钟总时长。导出包含实验隔离所需的重复构建，不能直接当作每次部署必需成本。

| Stage | Successful processes | Sum process seconds | Failed processes |
| --- | ---: | ---: | ---: |
| candidate calibration speed | 55 | 2177.80 | 0 |
| checkpoint export (repeated per experiment) | 338 | 9360.35 | 3 |
| validation NLL | 45 | 2053.43 | 3 |
| fresh allocation batch CPU process (all objectives) | 1 | 5.74 | 0 |
| independent test NLL | 45 | 1800.63 | 0 |
| action profiling process (includes init) | 1 | 26.34 | 2 |
| 12 feasible HAWQ allocations and main design CPU process | 1 | 4.31 | 0 |
| validation selection CPU process | 1 | 2.47 | 0 |

80项算子画像内部计时为23.08s，不包含模型/扩展初始化。10个正向MILP逐点solver中位数0.0618s，范围0.0265–0.1385s；这是一次冻结批次的描述性统计。

训练代理、SparseGPT canonical和HAWQ Hessian本轮复用既有结果；从零成本见run004/results/preparation_costs.json，例如HAWQ Hessian约1765.64s，不能将本轮缓存复用记为零成本复现。

运行时初始化包含Graph捕获，二者不可相加；采样NVML最大使用量不是连续测得的瞬时峰值。GPU权重存储已按实际storage去重；arena表示两阶段按层复用容量，host缓存另列。

| Main P+D method | Init s | Capture s (inside init) | Restore ms | Peak allocated GiB | GPU linear storage GiB | Host cache GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8.55 | 0.49 | 0.14 | 23.04 | 12.94 | 0.00 |
| Dense NVFP4 | 7.76 | 0.72 | 0.15 | 15.88 | 3.64 | 0.00 |
| W4A16 NVFP4 | 7.62 | 0.68 | 0.15 | 13.76 | 3.64 | 0.00 |
| W4A16 INT4 Marlin | 7.66 | 0.71 | 0.15 | 13.45 | 3.34 | 0.00 |
| Sparse BF16 | 8.64 | 0.85 | 0.14 | 17.37 | 7.28 | 0.00 |
| Sparse NVFP4 | 7.60 | 0.78 | 0.15 | 16.09 | 3.84 | 0.00 |
| INT8 W8A8 | 8.48 | 0.72 | 0.13 | 17.64 | 6.47 | 0.00 |
| SmoothQuant INT8 | 8.42 | 0.71 | 0.14 | 17.64 | 6.48 | 0.00 |
| FP8 W8A8 | 8.78 | 0.68 | 0.16 | 16.75 | 6.48 | 0.00 |
| SmoothQuant FP8 | 8.76 | 0.71 | 0.14 | 16.75 | 6.48 | 0.00 |
| HAWQ-v3 INT4, latency target=.80 BF16 (PTQ) | 9.22 | 0.75 | 0.16 | 19.70 | 8.53 | 0.00 |
| HAWQ-v3 FP4, latency target=.80 BF16 (PTQ) | 10.69 | 0.80 | 11.48 | 21.10 | 8.87 | 0.50 |
| HAWQ FP4 minimum-cost control, q=.07 | 12.74 | 0.79 | 47.20 | 15.54 | 4.99 | 3.80 |
| DASO, 5 candidates, fresh-profile DP | 15.44 | 0.90 | 96.82 | 17.26 | 5.02 | 8.66 |
| DASO + 8bit, fresh-profile DP | 15.24 | 0.87 | 86.65 | 17.67 | 7.12 | 7.68 |
| DASO + 8bit + exact/validation, epsilon=.01 | 16.33 | 1.65 | 71.56 | 18.71 | 6.48 | 10.11 |
| DASO + 8bit + exact/validation, epsilon=.04 | 16.47 | 1.02 | 94.62 | 18.76 | 6.52 | 10.55 |

全部policy和每轮原始开销见runtime_overhead.csv；恢复包含在cycle中，主表E2E沿用原协议不含请求前恢复。
