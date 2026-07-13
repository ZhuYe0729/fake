# Promising Scenarios and Kernel-Model Analysis

本文件从 broad-grid vLLM 实测结果中筛选“存在明显加速空间”的场景，并用 `fake/kernels/cutlass/cutlass_wrapper/modeling` 的 `KernelLatencyPredictor` 分析 fused Llama2-7B Linear 部分的速度。

## 筛选标准

- 只从 `dense_bf16` 为 OK 的可比场景中筛选。
- 至少一个压缩/异构方案的 vLLM 端到端 speedup `>= 1.70x`。
- 保留三类场景：长上下文生成、prefill-only/近似 prefill-only、以及中等输出长度场景。
- 这些场景不要求当前 hetero 实测已经最好；目标是找到压缩方案整体有明显收益、层异构优化值得发挥的区域。

## 预测假设

- 只预测 Linear kernel latency，不包含 attention、KV cache、scheduler、sampling 和 vLLM runtime overhead。
- 按 vLLM/Llama fused Linear 近似：`qkv(12288,4096)`、`o_proj(4096,4096)`、`gate_up(22016,4096)`、`down(4096,11008)`，每层 4 个 fused Linear，共 32 层。
- prefill 使用 `m=batch*input_seq`；decode 使用 `m=batch` 并重复 `output_seq` 次。
- `best_mixed` 是逐 phase/Linear shape 选择预测 latency 最低且满足 kernel 支持约束的 kernel；它是速度上界分析，不包含精度约束。
- 主表中空白的 single-method latency 表示该方法在 modeling 约束下无法完整覆盖该场景的 prefill+decode Linear 调用；具体原因见 CSV 的 `pred_*_status` 列。

## Selected Scenario Summary

| scenario | measured_best | measured_best_speedup | measured_hetero_speedup | pred_dense_bf16_ms | pred_dense_nvfp4_ms | pred_sparse_bf16_ms | pred_sparse_nvfp4_ms | pred_marlin_nvfp4_ms | pred_best_single | pred_best_mixed_ms | pred_best_mixed_speedup | mixed_vs_best_single | best_mixed_choices |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| b2_in16384_out128 | marlin_nvfp4 | 8.530 | 8.527 | 3479.751 | 3491.246 |  |  | 2825.649 | marlin_nvfp4:2825.649 | 1410.209 | 2.468 | 2.004 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |
| b2_in16384_out64 | hetero | 6.350 | 6.350 | 2784.638 | 2227.360 |  |  | 2492.340 | dense_nvfp4:2227.360 | 1086.968 | 2.562 | 2.049 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |
| b4_in16384_out128 | sparse_nvfp4 | 4.098 | 3.360 | 5701.400 | 4871.113 |  |  | 6520.909 | dense_nvfp4:4871.113 | 2163.803 | 2.635 | 2.251 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:4 |
| b4_in16384_out64 | sparse_nvfp4 | 3.870 | 3.150 | 4991.206 | 3523.026 |  |  | 6160.011 | dense_nvfp4:3523.026 | 1827.403 | 2.731 | 1.928 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:4 |
| b8_in16384_out128 | sparse_nvfp4 | 3.605 | 2.824 | 10232.731 | 7994.343 | 3465.003 |  | 20582.241 | sparse_bf16:3465.003 | 2643.215 | 3.871 | 1.311 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:4 |
| b8_in16384_out64 | sparse_nvfp4 | 3.473 | 2.717 | 9532.331 | 6742.284 | 2726.377 |  | 20248.368 | sparse_bf16:2726.377 | 2315.483 | 4.117 | 1.177 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:4 |
| b4_in1024_out1 | sparse_nvfp4 | 2.470 | 1.858 | 277.158 | 128.838 |  |  | 238.797 | dense_nvfp4:128.838 | 93.090 | 2.977 | 1.384 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |
| b2_in1024_out1 | sparse_nvfp4 | 2.455 | 1.945 | 148.428 | 76.595 |  |  | 123.424 | dense_nvfp4:76.595 | 52.331 | 2.836 | 1.464 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:1,sparse_nvfp4:3 |
| b4_in512_out1 | sparse_nvfp4 | 2.286 | 1.819 | 148.664 | 77.911 |  |  | 123.855 | dense_nvfp4:77.911 | 52.537 | 2.830 | 1.483 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:1,sparse_nvfp4:3 |
| b2_in4096_out1 | sparse_nvfp4 | 2.259 | 1.818 | 524.741 | 238.278 |  |  | 471.716 | dense_nvfp4:238.278 | 176.636 | 2.971 | 1.349 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |
| b1_in4096_out1 | sparse_nvfp4 | 2.251 | 1.796 | 277.011 | 127.189 |  |  | 238.262 | dense_nvfp4:127.189 | 92.938 | 2.981 | 1.369 | marlin_nvfp4:4,sparse_bf16:2,sparse_nvfp4:2 |
| b8_in512_out1 | sparse_nvfp4 | 2.187 | 1.409 | 277.005 | 127.338 | 144.872 |  | 238.375 | dense_nvfp4:127.338 | 92.954 | 2.980 | 1.370 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |
| b128_in256_out64 | sparse_bf16 | 1.776 | 1.719 | 2926.016 | 2239.540 | 1699.002 | 2001.896 | 2902.337 | sparse_bf16:1699.002 | 1350.223 | 2.167 | 1.258 | dense_bf16:1,marlin_nvfp4:1,sparse_bf16:4,sparse_nvfp4:2 |
| b32_in4096_out16 | sparse_nvfp4 | 1.734 | 1.366 | 9016.012 | 5808.041 | 2172.973 | 7469.994 | 20014.533 | sparse_bf16:2172.973 | 2086.433 | 4.321 | 1.041 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:4 |
| b256_in512_out16 | sparse_nvfp4 | 1.732 | 1.388 | 9157.439 | 5819.153 | 2218.967 | 7462.645 | 20211.251 | sparse_bf16:2218.967 | 2208.334 | 4.147 | 1.005 | marlin_nvfp4:1,sparse_bf16:7 |
| b64_in512_out16 | sparse_nvfp4 | 1.713 | 1.453 | 2283.598 | 1278.647 | 1113.578 | 1143.594 | 2283.328 | sparse_bf16:1113.578 | 886.966 | 2.575 | 1.255 | dense_bf16:1,marlin_nvfp4:3,sparse_bf16:2,sparse_nvfp4:2 |

## Interpretation

- vLLM 实测中最明显的加速集中在长上下文、小到中等 batch、较长输出，以及 output=1 的 prefill-dominant 场景。
- kernel 模型显示 best_mixed 通常会在 prefill 的大 `m` Linear 上偏向 sparse/dense NVFP4 类 kernel，在 decode 的小 `m` Linear 上偏向 marlin 或 dense BF16/NVFP4，避免单一方法在某些 phase 不占优。
- 若后续要把 best_mixed 变成真实策略，需要再叠加精度约束、fused module 约束以及 vLLM backend 的实际可用 kernel 约束。

## Files

- `promising_scenarios_modeling.csv`: 场景级汇总。
- `promising_scenarios_modeling_details.csv`: 每个场景、phase、fused Linear 的逐 kernel 预测和 best choice。
