## 2026-07-14 - End-to-end Llama2 paper-experiment runbook
- 开发目的：让迁移到新机器后的执行者能从一个给定模型完整跑通 baseline、代理建模、Pareto 闭环和论文级结果，而不依赖口头实验历史。
- 修改内容：新增 341 行中文运行手册，以 Llama2-7B-Chat 的真实路径、workload、runner 与结果目录为例，覆盖环境/extension smoke test、uniform 专用 quant-method baseline、WikiText 质量代理、roofline+E2E 速度校准、DP 求解、phase-hetero checkpoint、速度/NLL/下游任务闭环、汇总图表和审计清单。
- 影响文件：`artifacts/exports/vllm/LLAMA2_7B_CHAT_PAPER_EXPERIMENT_RUNBOOK.md`、`dev/plans/101_vllm_model_paper_experiment_runbook_plan.md`。
- 验证：检查了手册引用的核心 runner、solver、NLL、汇总文件均存在；Markdown 的 15 个本地链接均解析到存在路径。
