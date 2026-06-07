# Llama2 Warm Group Microbench Plan

## Summary
用更接近 warm E2E 的 group-level microbench 复测 `llama2-7b normal_02` 的关键 linear group，验证是否能修复 standalone manual decode 测量排序错误。

## Key Changes
- 新增 debug 脚本到 `artifacts/debug/005_llama2_warm_group_microbench/scripts/`。
- 对每个候选构造 32 个同 shape module，按 full-model group 顺序计时。
- timed loop 中不做逐次 `assert_finite`，不在候选间 `empty_cache`。
- 输出候选 ranking，并对比旧 standalone manual 和 full-model trace。

## Test Plan
- 跑关键 group：`mlp.down_proj`、`mlp.gate_proj`、`self_attn.o_proj`、`self_attn.q_proj`。
- 检查新 microbench 是否把 MLP hybrid、attention marlin 排到更接近 full-model trace 的位置。

## Assumptions
- 该脚本用于 debug 验证，不直接替换主 retest 逻辑。
