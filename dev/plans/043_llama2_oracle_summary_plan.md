# Llama2 Oracle Summary Plan

## Summary
整理 `llama2-7b` 在 `prefill_only`、`normal_01`、`normal_02` 三个场景下的 single、pred、oracle 结果到 `artifacts/results/main/003_llama2_oracle_summary/`，并生成 `summary.md`。

## Key Changes
- 汇总 6 种 single、pred、oracle 的 E2E 速度。
- 补齐第 6 种 single：`dense_nvfp4_prefill_marlin_decode`。
- 复制/生成 oracle 与 pred policy 对比。
- 说明 oracle 与 pred 的来源和估计方法。

## Test Plan
- 补跑缺失 E2E。
- 检查每个场景都有 8 类结果。
- 检查每个场景 policy 对比覆盖 7 个 linear group。
