# Llama2 vLLM Promising Max-Speed Policy Plan

## Summary

- 为 promising scenario 中每个场景补充一种 `max_speed_hetero` 策略。
- `max_speed_hetero` 在同一 vLLM fused 约束下，不使用 P024 精度预算，直接选择预测 latency 最小的层级压缩策略。
- 导出 unique max-speed checkpoint，实测 vLLM 速度和 full ARC-Challenge 精度，并把结果并入最终宽表。

## Key Steps

1. 求解 max-speed 策略：
   - 复用 fused shape 和 `KernelLatencyPredictor`。
   - 对每个 `(layer, fused_linear)` 直接选择该场景下预测 latency 最小的候选方法。
   - 输出场景到 unique policy 的映射和 policy JSON。

2. 导出模型：
   - 复用已有 per-policy hetero vLLM 导出脚本。
   - max-speed checkpoint 单独放在 `max_speed/checkpoints/`，避免混淆 P024 optimized checkpoint。

3. 测试速度和精度：
   - 速度使用 vLLM，与已有 promising benchmark 口径一致。
   - 精度使用 vLLM + lm-eval full ARC-Challenge 0-shot，每个 unique policy 只测一次。

4. 更新最终表格：
   - 在宽表中增加 `max_speed_hetero` speedup。
   - 增加 `max_speed_vs_best_single`。
   - 增加 `max_speed_acc_norm`。
   - 增加 `max_speed_policy`。
   - 在 policy details 中补充 max-speed 策略明细。

## Assumptions

- max-speed 策略不加精度预算，因此精度可能显著下降；这是用于上限速度对照。
- Marlin 仍不进入 hetero 策略求解候选，原因同 078/079：没有可信 Marlin-specific quality proxy 和 per-module fused 策略建模。
- single 方法速度和精度继续复用已有 vLLM/baseline 结果。
