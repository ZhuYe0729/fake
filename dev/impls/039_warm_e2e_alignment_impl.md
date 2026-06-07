## 2026-06-06 - Warm E2E aligned retest
- 开发目的：将 manual/pred 的离线估计语义对齐当前 warmed full-model E2E 测试，避免 prefill first materialization 进入 warm 估计。
- 修改内容：新增 `002_warm_e2e_aligned_policy_retest` 默认输出根；manual 加权公式改为使用 `prefill_steady_ms`；结果行增加 `timing_mode=warm_e2e_aligned`；补充 README/ANALYSIS。
- 影响文件：`scripts/run_main_hybrid_policy_retest.py`、`dev/plans/039_warm_e2e_alignment_plan.md`、`artifacts/results/main/002_warm_e2e_aligned_policy_retest/`。
- 后续注意：`llama2-7b normal_02` 中 pred 仍明显快于 manual；manual 只将 `mlp.up_proj` 选为 `dense_nvfp4->marlin_nvfp4`，说明 module-level manual 测量仍不能完全作为 full-model oracle。
