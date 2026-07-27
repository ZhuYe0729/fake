# Llama2-7B-Chat RTX Pro 6000 prefill-only 独立实验计划

## 目标

在 `artifacts/debug/064_llama2_pro6000_prefill_only/` 中建立并执行自包含的
Llama2-7B-Chat prefill-only 论文实验。实验固定 B=8、S=2048、O=1，覆盖
canonical sparse、72-policy NLL、Pro 6000 kernel profile、Pareto 真实闭环和
WikiText/WinoGrande/ARC/MMLU 质量评测。

## 实施约束

- 历史 033/046/054/060/063/104 目录只作为设计来源，不修改，也不作为运行时输出。
- 实验专用代码、输入副本、中间状态、日志、结果和验证报告全部位于 064。
- 原始模型、离线数据、CUTLASS wrapper 与 patched vLLM 是只读基础依赖。
- 正式速度在同一张独占 GPU 串行测量；NLL、local error 和任务可多卡并行。

## 成功标准

- 064 的脚本不运行时依赖历史 debug 结果目录。
- canonical sparse NVFP4 明确为 prequant-only，exporter 不 direct-prune 或重复量化。
- 72-policy NLL、当前 GPU predictor、solver 和 closure provenance 完整闭合。
- 正式点保留 1 warmup + 5 measured 原始记录，最终图表仅使用实测速度和质量。
