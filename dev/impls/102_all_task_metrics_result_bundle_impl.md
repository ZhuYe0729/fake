## 2026-07-14 - Expose all measured downstream metrics
- 开发目的：避免最终论文结果包只展示每个 prefill-decode 数据集的一个主指标，而隐藏已计算的辅助指标。
- 修改内容：Llama2 与 Llama3.1 聚合 CSV/Markdown 统一新增 CNN/DM BERTScore、DialogSum BERTScore、IWSLT ROUGE-L 三列；保留原有 CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU。两个结果包各新增三张相应的 speed-vs-metric Pareto 图。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/make_paper_result_bundle.py`、`artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/make_paper_result_bundle.py`、两个 `artifacts/exports/vllm/ours/*/pareto_summary/`。
- 验证：两脚本均通过 `py_compile` 并重新生成产物；Llama2 35 行和 Llama3.1 23 行均包含六个下游字段。Llama2 历史 i34/i36/i37/i38 只保存了当时计算的主指标，辅助列显示 `—` 且不会出现在辅助指标图中。

## 2026-07-15 - Complete Llama2 intermediate secondary metrics
- 开发目的：补齐 i34/i36/i37/i38 的 CNN/DM BERTScore、DialogSum BERTScore 和 IWSLT ROUGE-L，使全部 35 行 Llama2 结果均具有六项下游指标。
- 修改内容：复用已完成的 12 组 shard generation JSONL，在 GPU 5/6/7 并行执行标准 metrics-only merge/scoring；将完整 metrics 写入 `task_quality_intermediate/full_metrics/point_{34,36,37,38}/`，并让 paper bundle 聚合器读取该目录。速度和生成内容均未重测或改动。
- 影响文件：`artifacts/debug/036_llama2_prefill_decode_intermediate_points/task_quality_intermediate/full_metrics/`、`artifacts/debug/037_llama2_prefill_only_pareto/scripts/make_paper_result_bundle.py`、`artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/`。
- 验证：12/12 个 `metrics.json` 已生成；i34/i36/i37/i38 的三项新增字段均已写入最终 CSV/Markdown 和对应三张 Pareto 图。
