# Llama2 Warm E2E Gap Trace Plan

## Summary
通过完整模型内的 forward hook 实测 `llama2-7b normal_02` 下 linear group 的 prefill、first decode、steady decode 时间，定位 standalone manual module 测量与 full-model E2E ranking 的具体差异。

## Key Changes
- 新增 debug 脚本到 `artifacts/debug/004_llama2_warm_e2e_gap/scripts/`。
- 输出 manual/pred policy 的 full-model hook trace CSV 和聚合对比。
- 对照 `002_warm_e2e_aligned_policy_retest` 的 standalone manual candidates 和 policy 选择。

## Test Plan
- 跑脚本生成 manual/pred trace。
- 汇总每个 linear group 的 full-model prefill、decode_first、decode_steady 均值/总和。
- 写 `README.md` 说明结论。

## Assumptions
- hook trace 用于定位差异，绝对 E2E 数值会受到 hook 事件记录开销影响。
- 重点比较同一 trace 方法下 manual/pred 的相对 group 行为。
