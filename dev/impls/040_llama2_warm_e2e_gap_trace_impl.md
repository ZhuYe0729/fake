## 2026-06-06 - Full-model linear trace for Llama2 normal_02
- 开发目的：用真实完整模型内的 hook trace 定位 standalone manual module 测量与 full-model E2E ranking 的具体不一致。
- 修改内容：新增 debug trace 脚本，分别 trace `manual` 和 `pred` policy；输出 raw trace、按 decode step 聚合的 group projection，以及 standalone-vs-in-model 对照表。
- 影响文件：`artifacts/debug/004_llama2_warm_e2e_gap/`、`dev/plans/040_llama2_warm_e2e_gap_trace_plan.md`。
- 后续注意：trace 证明 manual 的 standalone scoring 在 `mlp.down_proj`、`mlp.gate_proj`、`self_attn.o_proj`、`self_attn.q_proj` 上与真实模型内 ranking 方向相反；pred 更接近 E2E 的原因是这些 group 的选择更符合 full-model trace。
