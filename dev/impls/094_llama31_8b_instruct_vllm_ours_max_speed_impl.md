# 094 Llama-3.1-8B-Instruct vLLM Ours Max-Speed Implementation

## 2026-07-13 - Initial implementation
- 开发目的：为 Llama-3.1-8B-Instruct 建立与 Llama2 max-speed 对齐的策略、导出、测速和 PMPD 评测流程。
- 修改内容：新增基于真实 Llama3.1 GQA fused QKV shape 的 predictor max-speed 策略生成器；新增独立的 phase checkpoint 导出、prefill-only 与 phase-switch 测速、PMPD shard 调度、汇总和使用说明。质量调度默认且校验仅允许 GPU 5、6、7。
- 验证：全部新增 Python 脚本通过 `py_compile`，Shell launcher 通过 `bash -n`；用本地 Llama3.1 配置完成 predictor smoke，确认生成 128-module policy，fused QKV shape 为 `[6144, 4096]`。当前执行环境的 NVIDIA driver 不可通信，故未在此环境启动导出、vLLM smoke、正式测速或 PMPD 作业。
- 影响文件：`artifacts/exports/vllm/ours/llama3.1-8b-instruct/`。
- 后续注意：所有 GPU 任务固定限制在 GPU 5、6、7。

## 2026-07-13 - Llama3.1 phase-speed runtime adjustments
- 开发目的：使 Llama3.1 phase-heterogeneous checkpoint 能在正式 prefill-decode workload 上完成 vLLM 测速。
- 修改内容：发现共享 runner 未限制 Llama3.1 的 131072 token config length，改用已有的 phase benchmark 通过 HF override 限制为 2128；随后确认 phase 双权重与 batch 16 prefill 在 `gpu_memory_utilization=0.9` 下 OOM，runner 默认改为 0.75，以为 NVFP4 activation packing 保留必要显存。
- 影响文件：`artifacts/exports/vllm/ours/llama3.1-8b-instruct/scripts/run_fresh_process_speed.sh`。
- 后续注意：测速仍保持加载后仅测 `generate` 的口径；0.75 仅改变 KV cache 容量，不改变 batch 或请求长度。

## 2026-07-13 - Formal speed completion
- 开发目的：完成两种 workload 的正式 max-speed vLLM 时延测量。
- 修改内容：prefill-only 使用 GPU 5 的 baseline-aligned 5 次 fresh process 测量；prefill-decode 使用 GPU 6、`max_model_len=2128`、`gpu_memory_utilization=0.75` 的 phase-runtime 1 warmup + 10 次测量。修复汇总脚本对 CSV 数值的类型转换。
- 验证：prefill-only E2E median 为 648.019 ms；prefill-decode TTFT median 为 1516.493 ms、E2E median 为 2539.234 ms、TPOT 为 12.946 ms。两种 checkpoint 均成功被 vLLM 加载并生成输出。
- 后续注意：尚待全量 PMPD 质量评测。

## 2026-07-13 - PMPD context-limit correction
- 开发目的：使 Llama3.1 checkpoint 能按 PMPD 的 `max_input_tokens=3840` 与 `max_new_tokens=256` 完成 vLLM KV cache 初始化。
- 修改内容：发现外部 PMPD evaluator 不提供 `max_model_len`，并会直接读取 checkpoint 的 131072 token config limit，导致所有 shard 在生成前失败。将两个实验 checkpoint 的服务上限固定为 4096，恰好覆盖评测最大序列长度且不修改权重或 RoPE 参数。
- 影响文件：两个 `max_speed/*/checkpoint/config.json`。
- 后续注意：需要以 5、6、7 三卡重新启动完整 PMPD shard 调度；此前没有有效生成结果。

## 2026-07-13 - Full PMPD quality completion
- 开发目的：完成两种 max-speed 策略在 CNN/DM 1000、DialogSum 1500、IWSLT 333 上的全量生成质量评测。
- 修改内容：通过 4096 context-limit smoke 后，使用 GPU 5、6、7 完成 5666 个样本生成、分片合并与六组 metrics 计算。
- 验证：六个 metrics.json 均已生成，所有数据集均无空预测；prefill-decode 的 CNN/DM、DialogSum、IWSLT 分别达到 Rouge-L/BERTScore `16.840/84.047`、`9.085/81.463`、Rouge-L/SacreBLEU `28.846/10.570`。prefill-only 的无约束 sparse 策略质量明显退化。
- 后续注意：对 prefill-only 的 predictor/runtime 失配分析另行处理。
