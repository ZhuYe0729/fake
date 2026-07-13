# 094 Llama-3.1-8B-Instruct vLLM Ours Max-Speed Plan

## Summary
- 为 Llama-3.1-8B-Instruct 求解 predictor-only、无质量约束的 phase-heterogeneous max-speed 策略，并在现有两个 vLLM workload 上测试速度与 PMPD 精度。
- 覆盖 `prefill_only`（`b=8,in=2048,out=1`）和 `prefill_decode`（`b=16,in=2048,out=80`）。
- 全部 GPU 工作只使用 GPU 5、6、7；不涉及超算调度配置。

## Key Changes
- 按 Llama3.1 GQA 的实际 fused QKV shape 求解每层策略，而不是复用 Llama2 的 `3 * hidden_size` 假设。
- 导出两个 phase-heterogeneous checkpoint，测速，并完成 CNN/DM 1000、DialogSum、IWSLT 的全量 PMPD 评测。
- 汇总 ours 与既有 Llama3.1 uniform baseline 的速度和质量结果。

## Verification
- 静态编译、策略/导出格式校验、每个 checkpoint 的 vLLM smoke generation。
- 速度使用既有 phase-runtime 正式口径；质量验证样本数和非空输出。

## Assumptions
- 仅实现 max-speed，不扩展 Pareto 或质量约束求解。
- 复用现有 Llama3.1 uniform baseline，不重跑其结果。
