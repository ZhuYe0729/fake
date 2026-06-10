## 2026-06-07 - Llama2 oracle summary
- 开发目的：整理 `llama2-7b` 三个场景下 single、pred、oracle 的 E2E 与 policy 对比结果。
- 修改内容：新增 `003_llama2_oracle_summary` 目录与汇总脚本；补跑三种场景的 `dense_nvfp4_prefill_marlin_decode` single E2E；生成 `comparison/*.csv` 和 `summary.md`。
- 影响文件：`artifacts/results/main/003_llama2_oracle_summary/`、`dev/plans/043_llama2_oracle_summary_plan.md`。
- 后续注意：`normal_02` oracle 与 pred policy 完全一致，E2E 行来自不同运行实例，数值差异应视为运行波动。
