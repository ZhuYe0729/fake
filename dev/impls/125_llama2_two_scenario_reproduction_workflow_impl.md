## 2026-07-21 - 可执行复现流程与当前平台验证

- 开发目的：将 Llama2-7B-Chat 的 prefill-only 与 B=8/S=2048/O=64 prefill-decode 最终论文实验整理为可迁移、可逐阶段验证的命令级流程。
- 修改内容：新增 063 runbook、统一配置、双环境/GPU/runtime 预检、clean bootstrap、阶段审计和四策略 smoke；为 054/056 关键入口补充环境变量路径覆盖；uniform/ours 明确统一使用 phase runtime 和 canonical SparseGPT state。
- 验证：Python/Bash 静态检查通过；当前 `cospaq`/`vllm` 环境与 8×RTX 5090 GPU 预检通过；bootstrap 的两组 72-policy 及 sample hash 通过；054/056/060 retained artifact 行数审计通过；fresh smoke 状态与细节见 063 `VALIDATION_REPORT.md`。
- 影响文件：`artifacts/debug/063_llama2_two_scenario_reproduction_workflow/`、054/056 的可迁移入口、共享 prefill evaluator path config、旧 runbook 的废弃提示。
- 后续注意：正式论文数值仍需按 runbook 跑 100-block NLL、1+5 独占测速和完整下游任务；smoke 仅验证链路。首次并发导出前必须串行预热，并排查中断遗留 extension lock。
