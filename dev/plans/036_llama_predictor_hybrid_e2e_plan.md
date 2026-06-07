# Llama Predictor Hybrid Full E2E Plan

## Summary
为 Llama-2-7B 与 Llama-3.1-8B 补充 predictor hybrid policy 的全模型替换接入，并运行真实端到端测试。结果保存到 `artifacts/results/benchmarks/hybrid/pred/`，与已有手动 hybrid 结果并列。

## Key Changes
- 新增 `fake/models/llama_kernels.py`：
  - 读取通用 offline hybrid policy JSON。
  - 通过 `select_compressible_modules(model, "llama")` 找到 Llama linear。
  - 支持 policy 中的 group suffix 名称匹配完整模块名。
  - 对 `dense_nvfp4 <-> marlin_nvfp4` 使用共享 canonical NVFP4 wrapper。
  - 对同 kernel 策略使用已有 backend module 构建函数。

- 新增 full E2E benchmark 脚本：
  - 加载本地 Llama 模型。
  - 应用 predictor hybrid policy。
  - 支持 `prefill_only` 与 `normal_01` 两个场景。
  - 输出真实 full model prefill/decode/e2e CSV。

## Test Plan
- 静态检查：`py_compile` 新增模块和脚本。
- GPU smoke：先跑 Llama-2-7B normal_01，再跑剩余模型/场景。
- 输出检查：确认 CSV 包含 predictor full E2E 与手动 hybrid 对比列。

## Assumptions
- 使用已有 policy：`artifacts/results/benchmarks/hybrid/pred/*_policy.json`。
- Llama policy 的 module name 是 group suffix，接入时按完整 module name 后缀匹配。
- 首版只补 Llama；Qwen3.5 已有 predictor_hybrid 接入。
