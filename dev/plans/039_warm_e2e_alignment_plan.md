# Warm E2E 对齐的 Manual/Pred 估计修正计划

## Summary
将 `manual` 和 `pred` 的离线估计语义对齐当前真实测试使用的 `warm_e2e`：正式计时前已经做过 prefill warmup，因此不计入 prefill backend 的首次格式转换/初始化；decode 阶段仍计入第一次在线切到 decode backend 时发生的 W4A16/Marlin 转换成本。

## Key Changes
- 保留当前 full-model E2E 流程为 `warm_e2e`。
- 新结果输出到 `artifacts/results/main/002_warm_e2e_aligned_policy_retest/`，保留旧 `001` 作为历史对照。
- manual 估计中 prefill 使用 `prefill_steady_ms`，不再把 `prefill_first_ms` 计入 `weighted_total_ms`。
- manual/pred 结果标记 `timing_mode=warm_e2e_aligned`，文档说明 linear summary 是 warm E2E 对齐估计。

## Test Plan
- `python -m py_compile scripts/run_main_hybrid_policy_retest.py`
- 跑 `llama2-7b normal_02` 的 single/manual/pred 全套 E2E。
- 检查 manual/pred policy 差异、linear summary 和 full E2E summary。

## Assumptions
- 正式目标是 `warm_e2e`，prefill backend 的首次 materialization/初始化不计入离线估计和 E2E。
- decode 阶段的在线 W4A16/Marlin materialization 仍计入。
- 先以 `llama2-7b normal_02` 验证，再决定是否扩展到其他模型和场景。
