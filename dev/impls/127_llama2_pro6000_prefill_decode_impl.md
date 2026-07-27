# 127 Llama2-7B-Chat / Pro 6000 / Prefill-Decode 实现记录

## 2026-07-22 - 完成独立 prefill-decode 复现实验

- 开发目的：在 2× NVIDIA RTX PRO 6000 Blackwell Server Edition 上，按 B=8、S=2048、O=64 协议完整复现 Llama2-7B-Chat prefill-decode 流程。
- 修改内容：在 `artifacts/debug/065_llama2_pro6000_prefill_decode` 内实现独立的预检、输入冻结、canonical 状态复制与哈希审计、72-policy teacher-forced NLL、双 phase 局部误差、质量拟合、8-shape kernel profile、动作审计、Pareto 求解、真实 closure、PMPD 三任务、汇总和全阶段校验。
- 协议修正：正式测速改为每个策略一个 fresh vLLM 进程，并在同一已加载引擎内执行 1 次 warmup + 5 次 measured；每次 raw 结果记录相同进程 ID 和独立 phase trace，validator 强制检查该条件。
- 数据处理：冻结 300×2112 WikiText 样本（SHA-256 `d807966dbb776dfcd43239bfc0cbaed40518d1bb1103b29e521af80f3219dfc3`）和历史 IWSLT 333 question IDs；共享模型与数据集缓存，不把下载缓存复制进实验目录。
- 主要结果：72/72 NLL、24/24 closure、140/140 PMPD shards、30/30 任务指标、24 行最终结果和 7 张图全部完成；`validation/all.json` 为 `ok: true`。
- 影响文件：`dev/plans/127_llama2_pro6000_prefill_decode_plan.md`、`dev/impls/127_llama2_pro6000_prefill_decode_impl.md`、`artifacts/debug/065_llama2_pro6000_prefill_decode/**`。
- 后续注意：论文引用应使用 `results/complete_results.csv` 和 closure 的正式 1+5 数据，不应使用 diagnostic-only smoke 数字；decode sparse-NVFP4 动作仍按审计结果明确禁用。
