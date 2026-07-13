# Llama2-7B vLLM 8 场景速度与精度 Baseline 计划

## Summary

- 在 `artifacts/exports/vllm/llama2_7b_018` 下补齐 selected 8 scenarios 的 vLLM benchmark、结果汇总和最终文档。
- 行包含 `dense_bf16`、`dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4`、`marlin_nvfp4`、层异构方法。
- 速度列为 8 个场景各自相对 `dense_bf16` 的 median speedup，外加 `avg_speedup`。
- 精度列主指标使用 full ARC-Challenge `acc_norm`；NLL delta 放入补充明细。
- 层异构精度只跑 3 个唯一策略，因为 8 个场景实际复用 3 套压缩模型。

## Key Changes

- 添加 selected-8 专用 vLLM 导出脚本，补齐 `uniform_marlin_nvfp4` 和 3 个 `hetero_strategy_*` checkpoint。
- 添加 selected-8 vLLM benchmark 脚本，按 handoff 文档固定 batch、input len、output tokens。
- 添加 vLLM/lm-eval ARC-Challenge 精度脚本，用实际导出的 checkpoint 评测，不在推理时做简单 Linear 替换。
- 添加汇总脚本，生成 CSV 和 Markdown 主表，并在表格下方注明层异构各场景采用的策略和精度。

## Test Plan

- 导出后检查每个 checkpoint 的 `config.json` 和 safetensors tensor 可被 vLLM 后端识别。
- 每个模型先跑小规模 sanity benchmark，确认 vLLM 可加载。
- 正式速度测试记录 warmup、timed iterations、median latency、tokens/s 和 speedup。
- 精度测试对缺失的 `marlin_nvfp4` 和 3 个 hetero checkpoint 使用 full ARC-Challenge。

## Assumptions

- speedup baseline 是同一场景下的原始 `dense_bf16` vLLM median latency。
- `single methods` 包含 handoff 文档里的 `marlin_nvfp4`。
- 层异构方法的精度按 3 个唯一压缩策略评测，然后映射回 8 个场景。
- 当前机器可直接用 GPU；如沙箱看不到 GPU，执行 benchmark/export 时申请非沙箱运行。
