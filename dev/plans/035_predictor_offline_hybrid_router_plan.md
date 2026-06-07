# 通用 Predictor-Driven Offline Hybrid 模块计划

## Summary
实现一个独立的离线 kernel 策略选择模块，而不是只写 Qwen 脚本。模块输入通用的 Linear shape/workload 描述，调用已有 `KernelLatencyPredictor` 和新增 `predict_conversion()`，输出每个 Linear 的固定策略。Qwen3.5 只作为一个接入方：负责枚举自己的 Linear、读取 policy、按 policy 构建 checkpoint。

## Key Changes
- 新增通用模块，例如 `fake/kernels/offline_hybrid_policy.py`：
  - 定义 `LinearShapeSpec(name, n, k, count=1)`、`ScenarioSpec(batch_size, input_tokens, output_tokens)`、`LayerPolicyDecision`、`HybridPolicy`。
  - 提供核心 API：`select_offline_hybrid_policy(linears, scenario, predictor, kernels=None) -> HybridPolicy`。
  - 模块不依赖 Qwen/Transformers/checkpoint，仅处理 shape、latency prediction、compatibility 和策略输出。

- 策略选择逻辑：
  - 对每层预测 `M_prefill=batch_size*input_tokens`、`M_decode=batch_size` 下所有候选 kernel latency。
  - 枚举合法策略，而不是先分别取 prefill/decode 最优：
    - 单 kernel 策略：prefill 和 decode 使用同一个 kernel，成本为 `prefill_ms + output_tokens * decode_ms`。
    - 兼容双 kernel 策略：仅允许 `dense_nvfp4 <-> marlin_nvfp4` 跨阶段组合。
  - 对不兼容的“prefill 最优 + decode 最优”不直接跳过，也不直接 fallback dense；模块会继续比较所有单 kernel 策略，选端到端最小者。
  - unsupported 或 prediction missing 的 kernel 只从对应候选中排除；如果某层无任何合法策略，再标记 `unselected` 并给出原因。

- NVFP4 转换开销：
  - 对 `dense_nvfp4/marlin_nvfp4` 双 kernel 策略，调用 `predictor.predict_conversion(n,k)`。
  - 成本加入 `canonical_to_cutlass + canonical_to_marlin` 中该策略实际需要的转换 latency。
  - 单独使用 `dense_nvfp4` 或 `marlin_nvfp4` 时只加入对应一次转换成本；非 NVFP4 canonical 路径不加转换成本。
  - v1 默认按一个离线场景冷启动计入一次转换成本，不做多请求摊销。

- Qwen3.5 接入：
  - 增加一个很薄的 Qwen adapter，用 `select_compressible_modules(model, "qwen3_5")` 生成 `LinearShapeSpec`。
  - 新增 `predictor_hybrid` method：读取通用 policy JSON，在 `fake/models/qwen3_5_kernels.py` 中按 policy 构建实际模块。
  - `dense_bf16` 在 policy 内保持通用名称，构建时映射到现有内部 `bf16`。
  - 现有 `shape_workload_hybrid`、manual hybrid 保持不变。

- 辅助 CLI 只作为模块包装：
  - 新增 `scripts/analyze_offline_hybrid_policy.py`，负责解析参数、加载模型/枚举 Linear、调用通用模块、写 JSON/CSV。
  - CLI 不承载核心选择逻辑，避免以后接别的模型时复制流程。

## Test Plan
- 单元/轻量测试：
  - 用 fake predictor 构造小表，验证单 kernel 统一选择能战胜不兼容的阶段最优组合。
  - 验证 `dense_nvfp4/marlin_nvfp4` 双策略会加入 `predict_conversion()` 成本。
  - 验证 unsupported kernel 被过滤，且无合法策略时有明确原因。
  - 验证输出 JSON round-trip 后 Qwen adapter 能读取同样的 per-layer backend。

- Qwen smoke：
  - 用 Qwen3.5 小变体生成 policy，确认每个 compressible Linear 都有 `selected_prefill_backend/selected_decode_backend/selected_total_ms`。
  - 用 `prepare_qwen3_5_kernel_checkpoint.py --method predictor_hybrid --policy-json ...` 构建 checkpoint。
  - 在 GPU 节点跑短 prefill+decode，确认 policy 生效并能 forward。

## Assumptions
- 首版优化目标是预测端到端 latency，不加入精度约束。
- 首版 workload 只建模标准 causal LM 的一次 prefill + `output_tokens` 次 decode；更复杂 workload 后续通过扩展 `ScenarioSpec`。
- `dense_nvfp4` 与 `marlin_nvfp4` 的兼容性仅来自共享 canonical NVFP4 weight；其他 kernel 必须 prefill/decode 同 kernel。
- 转换成本按每层每个离线场景计一次，后续如要长期服务摊销再加显式参数。
