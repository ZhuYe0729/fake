## 2026-07-15 - 建立 prefill-only 多任务质量评测框架

- 开发目的：在不污染主实验导出目录的前提下，为 Llama2-7B-Chat 与 Llama-3.1-8B-Instruct 的所有已汇总 prefill-only 策略补测 WikiText、Winogrande、ARC-Easy 与 MMLU。
- 修改内容：新建 `artifacts/debug/041_llama2_llama31_prefill_multi_task_eval`；从两个现有 `pareto_summary/summary.md` 自动生成 26 个策略清单；以 Transformers eager + lm-eval HFLM 动态注入各层 prefill 权重；提供跳过已完成结果的多 GPU 调度和与既有实测速率合并的汇总脚本。
- 影响文件：`041` 下 README、`scripts/build_manifest.py`、`scripts/evaluate_task.py`、`scripts/run_all.py`、`scripts/summarize.py`、`manifest/policies.json`、`summary/`。
- 验证：已完成 Python 编译、策略数（Llama2=14，Llama3.1=12）静态校验和待测汇总生成；按用户要求未启动 GPU 评测。
- 后续注意：GPU 空闲后先运行 README 中六策略 smoke，再运行全量；结果仅在 `041/results` 与 `041/summary` 中生成，审核后再决定是否写入 `artifacts/exports`。

## 2026-07-15 - Llama3.1 WikiText eager-attention OOM 修正

- 开发目的：修复 smoke 中仅 Llama3.1 WikiText PPL 的三项 OOM；其余 21/24 项（包括 Llama3.1 的三个判别任务）已成功。
- 修改内容：对 Llama3.1 的 WikiText HFLM 固定 `max_length=1024`，并把实际 context 写入每个结果 JSON；Llama2 保持 040 的默认 context。所有 Llama3.1 策略使用同一值，因此 PPL 横向比较口径一致。
- 后续注意：重跑三项失败的 WikiText 后才进入全量；不要将两个模型的绝对 PPL 直接横向比较。

## 2026-07-15 - 全量结果闭环与汇总列名修正

- 开发目的：完成 26 个策略 × 4 项任务的质量闭环，并确保汇总正确展示 WikiText PPL。
- 修改内容：全量 104 项中，Llama3.1 `ours_point_9` MMLU 首次因 HF 数据集服务器瞬时断连失败，单项重试后成功；修正汇总脚本将 `word_perplexity,none` 显示为 `wikitext_word_ppl`（原始 JSON 一直完整，问题仅限 Markdown 列名不一致）。
- 验证：最终 104/104 结果 JSON 完整，两个模型的 `full_*_prefill_only_multitask.md/.csv` 均无 pending 项。
- 后续注意：结果仍仅位于 `041` 调试目录；经人工确认后再汇入论文主结果表。
