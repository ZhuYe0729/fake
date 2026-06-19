## 2026-06-18 - Initial workflow scaffold
- 开发目的：为 FakeVLM 建立类似 Llama2 `018` 的 prefill-only 精度建模 + 速度建模 + Pareto policy + 实测验证流程。
- 修改内容：新增 `024_fakevlm_prefill_global_pareto` 实验目录、计划文件和脚本框架。
- 影响文件：`dev/plans/056_fakevlm_prefill_global_pareto_plan.md`，`dev/impls/056_fakevlm_prefill_global_pareto_impl.md`，`artifacts/debug/024_fakevlm_prefill_global_pareto/`。
- 后续注意：完整质量建模与验证需要在 GPU 节点运行；默认脚本提供 smoke/full 参数，不自动提交作业。

## 2026-06-19 - Local GPU smoke validation
- 开发目的：确认 `024` 流程能在本机 GPU 和 `cospaq` 环境下跑通。
- 修改内容：使用 GPU 7 跑通 local error 采集、stratified policy 生成、小样本质量验证、质量模型拟合、batch 16 cost/Pareto、dense/sparse policy prefill speed 验证、selected policy quality 验证和汇总。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/smoke/` 生成 smoke 结果。
- 后续注意：smoke 只覆盖 2 个模块和 2 个样本，结果仅用于验证流程，不代表最终 Pareto 结论。
