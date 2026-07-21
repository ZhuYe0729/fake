## 2026-07-15 - Launch real-vLLM prefill-only quality closure
- 开发目的：为 Llama3.1-8B-Instruct 替换不具备运行时量化语义的 HF 权重代理精度结果。
- 修改内容：新建 debug 043 的策略清单、七个 ours checkpoint 的可恢复串行物化脚本、真实 vLLM 审计/评测调度及汇总脚本；GPU1–5 启动五个 uniform 审计，GPU6 启动物化。
- 影响文件：`artifacts/debug/043_llama31_prefill_only_vllm_runtime_quality/`。
- 后续注意：只在运行时审计成功并且 export policy 与求解器 JSON 一致后执行全量任务；速度沿用既有 vLLM 实测结果。

## 2026-07-15 - Checkpoint storage constraint and resumed audit
- 开发目的：继续真实运行时闭环，同时避免磁盘空间不足产生不完整权重。
- 修改内容：五个 uniform 审计和 Llama2 sparse-BF16 补测均已通过；`ours_point_3/5/6` 的导出 policy 与 solver JSON 完全一致并进入审计。发现仅剩 52 GB 空间，`ours_point_8/9/11/13` 仅有元数据、无 `phase_hetero_policy.json`，不纳入结果。
- 后续注意：剩余四点采用物化→审计/评测→清理的流式执行；清理前保留策略 JSON、运行时元数据与任务结果。

## 2026-07-15 - Safe resume for incomplete exports
- 开发目的：补齐前次并发导出留下的四个不完整 checkpoint。
- 修改内容：物化脚本仅在输出目录缺少 `phase_hetero_policy.json` 时向 exporter 传递 `--force`，并在复用或新导出后逐字校验 policy JSON；完整、已验证 checkpoint 不会被覆盖。
- 后续注意：按单策略顺序物化、审计和全量评测，避免并发写入与部分目录阻塞恢复。

## 2026-07-15 - Preserve sparse policy export semantics
- 开发目的：使剩余含 sparse 方法的求解器策略能按与已有速度 checkpoint 相同的方式导出。
- 修改内容：物化脚本检测 default 与 module-level 方法；只要策略包含 `sparse_bf16` 或 `sparse_nvfp4`，自动传递 exporter 的 `--prune`。该行为与 `038` 的速度校准和 exports 的 checkpoint 准备脚本一致。
- 后续注意：稀疏策略的质量结果将对应真实 2:4 剪枝后的 vLLM checkpoint，而非未剪枝的权重代理。

## 2026-07-15 - Apply sparse workspace protection to heterogeneous policies
- 开发目的：防止含 sparse-BF16 子层的 ours 策略在 MMLU 变长请求中耗尽 matmul workspace。
- 修改内容：共享真实 vLLM evaluator 由标签判断改为解析 policy JSON；所有含 `sparse_bf16` 的 uniform/ours 策略均使用进程局部缓存上限 16、batch size 1 和 `max_num_seqs=1`。
- 后续注意：该保护只影响精度评测进程；默认内核缓存上限及所有已有速度结果不变。

## 2026-07-15 - Tighten sparse-BF16 evaluator workspace cap
- 开发目的：解决 `ours_point_11` 在 MMLU 全词表 prompt-logprob 临时张量上的 32-GB GPU OOM。
- 修改内容：真实精度评测进程中 sparse-BF16 matmul workspace cache 上限由 16 降至 4；仍保持 batch size 与 max-num-seqs 为 1。
- 影响文件：`artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py`。
- 后续注意：仅重跑尚未形成完整 five-task 结果的策略；此环境变量不改变默认内核缓存或既有速度结果。

## 2026-07-15 - Reserve prompt-logprob headroom for sparse policies
- 开发目的：为 MMLU 的全词表 FP32 prompt-logprob 临时张量预留确定的显存余量。
- 修改内容：含 sparse-BF16 的真实精度评测器将 vLLM `gpu_memory_utilization` 由 0.9 调至 0.8；该任务只使用单序列，不需要大 KV cache。
- 影响文件：`artifacts/debug/042_llama2_prefill_only_vllm_runtime_quality/scripts/evaluate_policy.py`。
- 后续注意：只影响尚未完成的精度评测进程与记录的运行时元数据；速度实验参数及历史结果完全不变。
