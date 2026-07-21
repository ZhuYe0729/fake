## 2026-07-15 - 建立真实 vLLM prefill-only 精度评测

- 开发目的：替换 HFLM 权重代理精度，真实执行 Llama2 uniform 与层异构 NVFP4 runtime。
- 修改内容：新建 `042_llama2_prefill_only_vllm_runtime_quality`，包含 14 个策略的 checkpoint/policy manifest、lm-eval VLLM likelihood evaluator、按策略可恢复调度和速度-质量汇总。
- 验证：完成 Python 编译与 14 个策略 checkpoint/policy 静态匹配；首次 audit 发现 `cospaq` 缺少真实 vLLM 的 pydantic 依赖，已改为已有 vLLM 测试使用的 `vllm` conda 环境，未产生任何有效质量结果。
- 后续注意：先完成六点 runtime audit，再启动全量；`041` 仅保留为失效的 weight-proxy 调试产物。

## 2026-07-15 - sparse BF16 runtime workspace OOM 修正

- 开发目的：完成全量过程中仅 sparse BF16 的 ARC-Easy workspace OOM 闭环。
- 修改内容：对 `sparse_bf16` 将 vLLM `max_num_seqs` 与 lm-eval likelihood batch 固定为 1；其余方法仍使用 4。模型、量化 kernel、token 序列和指标口径不变，仅降低评测并发与临时 workspace。
- 验证：全量首轮 13 个策略完整，sparse BF16 的 WikiText 与 Winogrande 已成功；重跑该策略以补齐 ARC-Easy、ARC-Challenge、MMLU。

## 2026-07-15 - Sparse BF16 variable-length workspace cache cap

- 开发目的：修复 sparse BF16 在 MMLU 大量不同 prompt 长度下累计 cuSPARSELt workspace 后的 OOM。
- 修改内容：为 kernel cache 增加默认不变的环境变量上限；本评测仅 sparse BF16 使用 16-entry LRU，并使用独立 extension build 目录触发重编译。
- 后续注意：此项仅改变 workspace 驻留策略，所有数值、checkpoint 和任务输入保持不变；速度主结果不受默认配置影响。
