## 2026-07-07 - 初始化 promising scenario retest

- 开发目的：为 broad-grid 筛选出的高潜力场景重新求解 P024 精度预算下的场景最优层异构策略，并准备 vLLM 速度复测。
- 修改内容：创建计划文件，明确 single 方法复用 broad-grid，hetero 使用 018 精度代理和 P024 budget 求解。
- 影响文件：`dev/plans/078_llama2_vllm_promising_policy_retest_plan.md`，`dev/impls/078_llama2_vllm_promising_policy_retest_impl.md`。
- 后续注意：Marlin 暂不进入质量约束求解候选；后续若建立 Marlin quality proxy 可重新打开。

## 2026-07-07 - 完成 promising scenario optimized hetero 复测

- 开发目的：针对筛选出的高潜力场景，按 P024 精度预算重新求解层异构压缩策略并实测 vLLM 速度。
- 修改内容：新增 fused-shape 速度预测求解脚本、8 个 unique policy 的 vLLM checkpoint 导出脚本、单卡/多卡 benchmark 脚本和结果汇总脚本；导出并测试 16 个场景的 `optimized_hetero`。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/` 下的 `scripts/`、`policies/`、`checkpoints/`、`benchmarks/`、`summary/`。
- 后续注意：当前 optimizer 未纳入 Marlin 候选；summary 中 best single 仍包含已测 Marlin，因此 `opt_vs_best_single` 是与完整 single baseline 比较。

## 2026-07-07 - 补充 method-wide speedup 表

- 开发目的：提供按场景展开的最终速度表，方便横向比较 single 方法、原 hetero 和重新求解的 optimized hetero。
- 修改内容：在 summary 脚本中新增 speedup 宽表输出，speedup 统一为同场景 dense bf16 median latency 除以目标方法 median latency。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/scripts/summarize_promising_policy_retest.py`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/summary/promising_policy_retest_speedup_wide.csv`，`artifacts/exports/vllm/llama2_7b_018/promising_policy_retest/summary/promising_policy_retest_speedup_wide.md`。
- 后续注意：该表中的 single 方法和 `original_hetero` 来自 broad-grid 的 vLLM 实测；`optimized_hetero` 来自本次重新导出 checkpoint 后的 vLLM 实测。
