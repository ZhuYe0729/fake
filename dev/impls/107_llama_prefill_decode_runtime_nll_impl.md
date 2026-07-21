## 2026-07-15 - Real vLLM teacher-forced decode-NLL mechanism
- 开发目的：替代历史 HF/权重代理的 prefill-decoding NLL，同时不修改任何旧产物。
- 修改内容：新建 debug 044；通过 vLLM V1 注册式 logits processor，在每个 decode step 捕获目标 token 的原始 logprob 后强制该 token 回填 KV cache；实现可复用的 2048-token prefill + 80-token decode runner。
- 验证：Llama2 BF16 单块记录 80 个 token（avg NLL 0.978279）；Llama3 phase-hetero point_002 单块记录 80 个 token（avg NLL 1.890872）。
- 后续注意：批量异构结果必须开启 trace 并验证 `enter_decode` / `apply_decode`；输出仅写入 debug 044。

## 2026-07-15 - Enforce phase-boundary evidence for full NLL runs
- 开发目的：防止 phase-heterogeneous NLL 测试仅加载异构 checkpoint、却未真正经历 prefill→decode 切换。
- 修改内容：runner 写出 trace 后强制校验 `enter_decode` 与 `apply_decode` 事件，并将完整事件计数写进运行时元数据。
- 影响文件：`artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/evaluate_runtime_decode_nll.py`。
- 后续注意：已完成的 point_002 有独立 trace 证明；后续新结果将自动携带同等强度的校验。

## 2026-07-16 - Correct phase trace event aggregation
- 开发目的：修正 point_006 已执行完推理、但因 trace JSON 的列表结构被误读而未写最终结果的问题。
- 修改内容：从 `trace` 事件列表汇总事件计数，而非读取不存在的 `events` 字段；不修改既有结果或不完整 trace。
- 后续注意：以带 `retry` 名称的新输出重新运行 point_006，保证旧调试证据仍可追溯。

## 2026-07-16 - Stream non-retained Pareto checkpoints
- 开发目的：补齐正式 Pareto 点的真实 NLL，同时在剩余约 23 GB 磁盘空间内不保留重复 checkpoint。
- 修改内容：新增单点工具：严格校验 source/export policy 一致后，临时导出（含 sparse 时 2:4 prune）、运行 phase-aware teacher-forced NLL、最后仅删除该临时 checkpoint。
- 影响文件：`artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py`。
- 后续注意：结果 JSON、capture、trace 与求解器 policy JSON 均永久保留；历史及既有 debug 结果不覆盖。

## 2026-07-16 - Queue remaining formal prefill-decoding points
- 开发目的：避免串行流式补测时 GPU 空闲，并完整覆盖未持久化 checkpoint 的正式 Pareto 行。
- 修改内容：加入严格 no-clobber 队列；跳过 policy-identical dense-BF16 point 000 和已有真实结果的点，覆盖 Llama2 points 1/2/4–10 与 Llama3.1 points 1/5/7/9。
- 影响文件：`artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/run_remaining_formal_queue.py`。
- 后续注意：队列状态写入 `run_state/formal_queue.json`；任何单点失败会停止队列，避免静默跳过。

## 2026-07-16 - Separate real-vLLM NLL report
- 开发目的：提供可审计的完整新 NLL 清单，同时不混入或覆盖历史 proxy NLL。
- 修改内容：新增汇总脚本，输出 CSV 与 Markdown，记录模型、uniform/正式 Pareto/max-speed 家族、NLL、PPL、phase trace 路径与事件元数据。
- 影响文件：`artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/summarize.py`。
- 后续注意：表中所有点固定为 32×80 个真实 decode token；历史报告仍保留在原位置。
