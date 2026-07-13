## 2026-07-11 - Predicted WikiText Pareto frontier
- 开发目的：将当前最佳 WikiText pooled 精度代理与 raw kernel speed predictor 组合成可复现的 debug Pareto solver。
- 修改内容：新增 034 debug solver；在每个场景重新拟合并冻结 54-policy proxy，构建 128 个 vLLM fused module 的 runtime-legal phase candidate，使用离散 quality-budget DP 输出 phase-policy JSON、候选 CSV 与预测曲线。
- 验证：prefill-only 输出 17 个唯一预测点，端点从 dense 1004.27ms 到 357.99ms raw linear latency（2.81x）；prefill-decode 输出 12 个唯一预测点，端点从 2970.74ms 到 1461.46ms（2.03x）。
- 后续注意：速度是 raw predictor linear latency，质量是 WikiText proxy contribution；这些不是已验证的 vLLM E2E 或真实下游曲线。
