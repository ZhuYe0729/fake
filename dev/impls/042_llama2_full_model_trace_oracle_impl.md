## 2026-06-07 - Llama2 full-model trace oracle
- 开发目的：从真实 full-model 推理出发，为 `llama2-7b normal_02` 构建并验证 oracle policy。
- 修改内容：新增 full-model method trace、trace oracle policy 构建、oracle E2E 验证，以及 k/q/v attention ablation refinement 脚本；跑完 6 种 method trace、初始 trace oracle E2E 和 8 个 attention ablation E2E。
- 影响文件：`artifacts/debug/006_llama2_full_model_trace_oracle/`、`dev/plans/042_llama2_full_model_trace_oracle_plan.md`。
- 后续注意：初始 hook trace oracle 会误选 k/q/v 为 dense bf16；直接 full-model ablation 后的 refined oracle 与 pred policy 完全一致。
