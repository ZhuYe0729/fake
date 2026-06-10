# Llama2 Full-Model Trace Oracle Plan

## Summary
从真实 full-model 推理出发，为 `llama2-7b normal_02` 构建 debug 级 oracle。分别跑候选 policy 的完整模型 forward trace，记录每个可压缩 linear module 的真实模型内耗时，再基于 group-level 投影延迟选择 oracle policy 并跑无 hook E2E 验证。

## Key Changes
- 新增 debug 脚本到 `artifacts/debug/006_llama2_full_model_trace_oracle/scripts/`。
- 支持 trace `dense_bf16`、`sparse_bf16`、`dense_nvfp4`、`sparse_nvfp4`、`marlin_nvfp4`、`dense_nvfp4_prefill_marlin_decode`。
- 生成 oracle policy、oracle E2E 结果和 README 分析。

## Test Plan
- 编译新增脚本。
- 跑 6 个 method 的 full-model module trace。
- 构建 oracle policy。
- 跑 oracle 无 hook E2E，与 `002_warm_e2e_aligned_policy_retest` 结果对比。

## Assumptions
- 第一轮只覆盖 `llama2-7b normal_02`。
- timing 语义为 `warm_e2e_aligned`。
- 默认 trace 32 个 decode step，使用 first decode + steady decode 投影到 256 token。
