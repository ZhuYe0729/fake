## 2026-06-06 - Warm group microbench validation
- 开发目的：验证更接近 warm E2E 的 group-level microbench 是否能修复 standalone manual decode 排序错误。
- 修改内容：新增 debug microbench 脚本，按同 shape 32 个 module 顺序执行，去除 timed loop 内逐次 finite check，避免 per-candidate empty_cache；输出 ranking。
- 影响文件：`artifacts/debug/005_llama2_warm_group_microbench/`、`dev/plans/041_llama2_warm_group_microbench_plan.md`。
- 后续注意：新 microbench 修复了 `mlp.gate_proj`、`mlp.up_proj`、`self_attn.o_proj`、`self_attn.q_proj` 的排序方向；`mlp.down_proj` 仍与 full-model trace 有小幅分歧。
