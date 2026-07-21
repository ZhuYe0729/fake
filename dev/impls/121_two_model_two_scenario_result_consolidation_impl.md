# 121 Two-model, two-scenario result consolidation implementation

## 2026-07-20 - Initial consolidation
- 开发目的：将 Llama2-7B-Chat 与 Llama3.1-8B-Instruct 的 prefill-only、prefill-decode 已保留实验结果收敛到一个易读、可追溯的 debug bundle。
- 修改内容：新增 `artifacts/debug/060_two_model_two_scenario_result_consolidation/`；每个模型/场景包含 `data/`、`policies/`、`results/`、`figures/` 和 `summary.md`。复制 compact CSV、policy JSON、既有 Pareto 图和源 summary，不复制 checkpoint/大日志；decode 表将 closure speed 与长格式三任务结果 pivot 为单行 policy 表。
- 验证结果：四个 `complete_results.csv` 分别含 Llama2 prefill 15 行、Llama2 decode 10 行、Llama3 prefill 15 行、Llama3 decode 17 行（均含表头之外的 retained policy rows）。修复了 Llama3 decode uniform `p00`–`p04` 仅出现在 task CSV、初版未带入 speedup 的拼接问题。
- 后续注意：该目录是只读汇总；缺失字段代表原实验没有保留相应测量，不能补推。Llama3 prefill high-sparsity speed-model 诊断仍在 debug 059，未在此篡改实测结果。

## 2026-07-20 - Compact one-policy-per-row source summaries
- 开发目的：使汇总目录中的 `results/source_summary.md` 便于论文结果审阅，而非按数据集重复同一 policy。
- 修改内容：新增 summary 重建脚本；四个场景均按 “uniform references → ours/solved policy points” 顺序输出，每个 policy 一行，原始逐数据集长表继续保留在 `data/task_summary_long.csv`。
- 验证结果：Llama3 decode 的 `p00`–`p04` uniform 行置顶，随后为 `point_000`–`point_011`；Llama2 decode 置顶 canonical dense-BF16 anchor `b8o64000`，随后为 `b8o64001`–`b8o64009`。无独立 retained uniform 行的字段明确保持为空。
