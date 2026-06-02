# Qwen3.5 Shape + Workload Hybrid 测试场景计划

## Summary
目标是证明一个二维 hybrid 策略：根据每个 `Linear` 的 shape `(N, K)` 和当前输入负载 `M` 共同选择最快轻量化路径，而不是只按 prefill/decode 切换。策略函数为 `policy(M, N, K) -> {dense_bf16, marlin_nvfp4, sparse_bf16, sparse_nvfp4}`，先只优化速度，不考虑精度。

## Key Changes
- 新增 Qwen3.5 per-linear hybrid runtime：
  - 对每个 compressible `Linear` 记录 `name, N=out_features, K=in_features`。
  - 每次 forward 根据输入 activation 的实际 `M = batch * tokens` 决定该层使用哪个 packed backend。
  - 同一个权重允许同时保存多种 packed 表示，runtime 只切换 kernel/activation quantization，不重新 pack。
- 策略采用 benchmark-derived lookup，而不是手写 prefill/decode 规则：
  - 对已有 `5_kernel_comprehensive` 结果按 `(M, N, K)` 或最近 bucket 查询最快 kernel。
  - 若 shape 未覆盖，用保守规则：小 `N/K <= 1024` 倾向 `dense_bf16`；小 `M <= 16` 倾向 `marlin_nvfp4`；大 `M >= 128` 且大 `N/K` 倾向 `sparse_nvfp4`。
  - 记录每层每 workload 的最终选择，生成 policy trace，论文中直接展示 hybrid 不是单一阶段切换。
- 实验方法分组：
  - 单一路径 baseline：`dense`、`marlin_nvfp4`、`sparse_nvfp4`。
  - 一维 hybrid：只按 workload 切换，prefill 用 `sparse_nvfp4`，decode 用 `marlin_nvfp4`。
  - 二维 hybrid：按 `(M, N, K)` 同时决策，小投影/不利 shape 可退回 dense 或 sparse_bf16。

## Test Scenarios
主论文优先选择能同时体现两种 hybrid 的场景：

| 场景 | batch | input | output | 预期展示点 |
|---|---:|---:|---:|---|
| Long-context RAG | 1 | 8192 | 512 | 同一大 Linear：prefill 用 WA，decode 用 W-only |
| Agent long generation | 1 | 4096 | 1024 | decode 占比高，证明 workload hybrid 必要 |
| Batched serving | 4 | 4096 | 512 | decode `M=4` 仍适合 W-only，prefill `M=16384` 适合 WA |
| Mixed-shape stress | 1 | 2048 | 512 | 同一阶段内，不同 Linear 因 N/K 不同选择不同 kernel |
| Boundary model | 1 | 8192 | 512 | 用 0.8B/2B 展示小 shape 不适合全量压缩，必须 shape-aware |

模型分层：
- 主结果：Qwen3.5-9B、Qwen3.5-27B。它们大多数 MLP/attention projection 位于大 shape 区域，最有利于展示 hybrid。
- 次结果：Qwen3.5-4B。用于显示策略从中型模型开始有效。
- 边界分析：Qwen3.5-0.8B、Qwen3.5-2B。重点展示纯 `sparse_nvfp4` 或纯 `marlin_nvfp4` 可能输，而 shape-aware hybrid 可以避免退化。

## Figures And Claims
- Figure 1：`policy(M,N,K)` decision map，横轴 `M`，纵轴按 Qwen3.5 linear shape 分组，颜色表示 kernel。
- Figure 2：端到端 latency/speedup，对比 dense、单一路径、一维 hybrid、二维 hybrid。
- Figure 3：per-layer backend trace，展示同一 workload 内 `mlp.gate/up/down`、attention projection、小 projection 选择不同 kernel。
- 核心结论：
  - 只按阶段切换不够，因为不同 Linear shape 的最优 kernel 不一致。
  - 只按 shape 固定选择也不够，因为同一个 Linear 在 prefill 和 decode 下的最优 kernel 会变化。
  - 二维 hybrid 同时利用 workload locality 和 shape heterogeneity，因此比单一路径和一维 hybrid 更稳。

## Assumptions
- 首版只做 text-only Qwen3.5。
- 首版只优化速度，不纳入精度约束。
- 实现阶段后续修改记录追加到 `dev/impls/034_qwen35_shape_workload_hybrid_impl.md`。
