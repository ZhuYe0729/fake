## 2026-07-14 - Unified Llama2 measured-result bundle
- 开发目的：将 Llama2-7B-Chat 的两场景全部实测 ours 策略、全部 uniform 方法和 max-speed 端点整理为与 Llama-3.1 一致的论文筛选包，而不是只保留单个代表点。
- 修改内容：新增确定性聚合/绘图脚本，导出 35 行全量 CSV 和 Markdown 总表，以及 ARC、CNN/DM、DialogSum、IWSLT、WikiText NLL 五张图。prefill-only 包含 9 个 ours 与 5 个 uniform 点；prefill-decode 包含 12 个 formal ours、4 个 screened intermediate 和 5 个 uniform 点。推荐仅作候选标记：prefill-only 为 008/012/013/016，prefill-decode 为 003/007/i38/011。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/make_paper_result_bundle.py`、`artifacts/exports/vllm/ours/llama2-7b-chat/pareto_summary/`、`dev/plans/100_llama2_paper_result_bundle_plan.md`。
- 验证：脚本通过 `py_compile` 并成功生成 35 条数据行；max-speed endpoint `ours_point_016` 与 `point_011` 已明确显示。中间 i34/i36/i37/i38 使用 stall-screened timing，表和图中以 `*` 区分，不能与 formal closure 做细粒度时延结论。
