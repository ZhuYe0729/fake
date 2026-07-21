## 2026-07-16 - Create corrected runtime-quality consolidation bundle
- 开发目的：在不修改 exports 或历史汇总的前提下，让用户直接查看两个模型、两个场景的实测速率与修正后真实 vLLM 精度。
- 修改内容：新建 debug 045 合并器，读取既有 speed closure、debug 042/043 prefill 五任务结果与 debug 044 teacher-forced NLL，生成 CSV、可直接阅读的 Markdown 表和四张 speed-quality 图。
- 影响文件：`artifacts/debug/045_runtime_quality_consolidation/`、`dev/plans/108_runtime_quality_result_consolidation_plan.md`。
- 后续注意：Llama2 的 i34/i36/i37/i38 为历史中间速度点，明确标为未重测 NLL；不会被画入修正后的 NLL 曲线。

## 2026-07-16 - Add prefill-decoding downstream task metrics
- 开发目的：让 debug 045 同时展示 prefill-decoding 的真实 NLL 与此前已测的三组生成任务结果。
- 修改内容：合并 CNN/DM（ROUGE-L、BERTScore）、DialogSum（ROUGE-L、BERTScore）和 IWSLT（ROUGE-L、BLEU）到 CSV 与 Markdown 独立表中。
- 影响文件：`artifacts/debug/045_runtime_quality_consolidation/scripts/build_summary.py`、`report/prefill_decode_downstream_tasks.csv`。
- 后续注意：这些是既有真实 vLLM generation-task 实测结果；只替换过 proxy 的 NLL，不应把它们标为新 NLL 评测值。

## 2026-07-16 - Generate per-metric Pareto views
- 开发目的：让每个 prefill-only 和 prefill-decoding 数据集/指标都有独立、量纲一致的速度-质量图。
- 修改内容：为每个模型生成 5 张 prefill-only 图（PPL、WinoGrande、ARC-Easy、ARC-Challenge、MMLU）与 7 张 prefill-decoding 图（NLL 和三组生成任务的两项指标）。
- 影响文件：`artifacts/debug/045_runtime_quality_consolidation/report/pareto/`。
- 后续注意：NLL 图排除没有新 NLL 的 Llama2 intermediate 行；生成任务图保留这些已有真实 vLLM 任务实测点，并以灰色三角区分。
