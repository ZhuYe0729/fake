# Llama2 vLLM Promising Scenario Policy Retest Plan

## Summary

- 对 `broad_grid_vllm/summary/promising_scenarios_modeling.csv` 中筛选出的高潜力场景，重新求解场景相关的层异构策略并做 vLLM 速度测试。
- single 方法速度复用 broad-grid 已测结果，不重复测试。
- hetero 策略按 018 的精度建模口径，在 P024 quality budget 内求每个场景最快 policy。

## Key Steps

1. 构建 per-scenario policy 求解脚本：
   - 输入 018 `module_method_candidates.csv` 的质量成本。
   - 使用 fused vLLM Linear 约束：`q/k/v -> qkv` 同方法，`gate/up -> gate_up` 同方法。
   - 使用 `KernelLatencyPredictor` 根据每个场景的 prefill/decode workload 预测 fused Linear latency。
   - 在 P024 quality budget 内求最小 latency policy，并对 policy 去重。

2. 扩展 vLLM hetero 导出：
   - 支持 per-layer/per-fused-linear method map，而不是只支持全模型 4 类 fused Linear 的统一 assignment。
   - 只导出唯一 policy，避免重复 checkpoint。

3. 测试速度：
   - 对筛选场景只测试新 hetero policy。
   - single 方法直接 join broad-grid latency/speedup。
   - 每张 GPU 同时只跑一个 vLLM 进程。

4. 汇总文档：
   - 输出每个场景的 single best、原 broad-grid hetero、new optimized hetero latency/speedup。
   - 记录每个场景对应 policy、质量预算、预测 latency、method counts。

## Assumptions

- 使用 P024 作为 balanced 精度约束：`quality_budget=0.2958401463404409`。
- Marlin 不进入精度约束求解候选，因为 018 README 明确没有可信 Marlin-specific quality proxy；Marlin single baseline 仍保留在对比表中。
- 速度测试使用已有 vLLM backend 和已导出的 uniform/bf16 baseline，不重新测试精度。
