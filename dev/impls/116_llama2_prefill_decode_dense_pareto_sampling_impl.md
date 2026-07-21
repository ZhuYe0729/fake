# 116 Llama2 prefill-decode dense Pareto sampling

## 2026-07-19 - 定位并隔离 V1 chunked-prefill 速度跳变
- 开发目的：解释原模型求解出的相邻策略为何从约 1.36x 突变至约 1.79x，而不是形成连续可用的 Pareto 速度段。
- 根因证据：`dense_006` 与仅将第 1、31 层 `mlp.down_proj` 改为 sparse-BF16 的对照策略 decode method_map 完全一致；但 vLLM 日志分别仅分配 `30,368` 和 `35,312` 个 KV token。前者少于固定输入的 `16*2048=32,768`，V1 因而分块处理 prefill；后者可一次完成。V1 的 `_set_default_args` 会对非 pooling 模型无条件设定 `enable_chunked_prefill=True`，因此 runner 传入的 `False` 实际无效。
- 修改内容：为 phase-hetero benchmark 增加 `--kv-cache-memory-bytes`，并由 prefill-decode speed runner 通过 `KV_CACHE_MEMORY_BYTES` 透传；metadata 同时记录该固定容量。该参数只固定 runtime 资源/调度条件，不改动模型、策略、权重或 kernel。
- 影响文件：`artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py`、`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/run_pareto_speed_point.sh`。
- 验证：Python 语法与 shell 语法检查通过。待 GPU 复测验证固定不少于 32,768-token KV cache 后两个对照策略的 E2E 差异回到逐层 kernel 的合理量级；此前的跳变速度结果不得进入正式 Pareto 图。

## 2026-07-19 - 禁止 V1 chunking 后发现 request-wave phase bug
- 开发目的：验证是否可仅通过禁止 V1 chunked prefill 恢复 phase-heterogeneous 策略的正确语义。
- 修改内容：修正 vLLM `engine/arg_utils.py` 的 V1 默认逻辑：仅当调用方未指定时才启用 chunked prefill，保留 benchmark 的显式 `enable_chunked_prefill=False`；runner 固定使用 V1。benchmark single-process 输出补建父目录；诊断 profiler 记录每个 Linear 调用的 phase/token-count。
- 验证：日志确认 `chunked_prefill_enabled=False`，dense-BF16 完成 B=16/S=2048/O=80 E2E smoke（4959 ms）。但其可用 KV cache 仅为 19,840 token（0.80 profile）；trace 显示第一波 `prefill/tokens=18432` 后，runtime 进入 decode，随后第二波 `decode/tokens=14336`。因此 V1 将 16 个请求拆为 9+7 个完整 prefill 波次，当前全局 phase 状态错误地将第二波 prompt 当作 decode。
- 后续注意：固定低 KV 让所有策略 chunk 的方案不可用于正式结果；仅禁止 chunk 也不足够。必须在 phase runtime 中支持 request-wave 的 prefill 重入（并在可能混合 prefill/decode token 时明确拒绝或拆分），才能有效测试 B=16 场景。

## 2026-07-19 - FP8 KV cache 恢复单波 B=16 phase 语义
- 开发目的：在不增加 request-wave 重入机制的情况下，让 16 个请求一次完整 prefill、再共同 decode。
- 修改内容：benchmark/profiler 支持显式 `kv_cache_dtype`；以 vLLM 原生 `fp8` KV cache 搭配已修复的 no-chunk V1 运行。FP8 将 dense-BF16 的动态 KV capacity 从 BF16 的 26,256（0.90）/19,840（0.80）提升至 52,528（0.90）/39,680（0.80）tokens。
- 验证：B=16/S=2048/O=80 trace 的有效请求记录为 `prefill/tokens=32768` 128 次、`decode/tokens=16` 10112 次，未出现第二个 prefill wave；因此全局 phase 切换与场景语义一致。单次 fresh smoke 完成，profile warm E2E 为 4022.7 ms。
- 后续注意：vLLM 对 FP8 KV 提示未给 scaling factor 时可能有精度下降。若将其作为正式统一 runner，所有 uniform/ours 的速度与任务质量必须使用同一 FP8 KV 设置，并先以 dense-BF16 和代表性异构点验证 NLL/任务指标影响。

## 2026-07-19 - FP8 KV cache quality gate
- 开发目的：验证 FP8 KV-cache 方案在采用前没有明显的下游任务质量回退。
- 修改内容：PMPD vLLM evaluator 新增 `--kv-cache-dtype {auto,fp8}`，并由既有 shard runner 透传且写入 `run_config.json`。在 V1、no-chunk、相同 greedy sampling 下，对 20 条 IWSLT 样本分别比较 BF16 KV 与 FP8 KV。覆盖 dense-BF16 和代表性混合 `point_006`（prefill 为 110 dense-NVFP4 + 18 sparse-NVFP4；decode 为 96 w4a16 + 32 BF16）。
- 验证：dense-BF16 SacreBLEU 为 19.24（BF16 KV）/21.39（FP8 KV）；point_006 为 16.62/19.15。20 样本仅能作为 no-obvious-regression gate，不能据此声称提升。greedy 输出逐字一致率分别为 2/20 与 0/20，说明 FP8 不是数值等价替换，必须定义成独立且统一的正式 runtime 口径。
- 影响文件：`/home/agent/wja/project/my/cospaq/test/vllm/artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py`、`artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/scripts/run_task_quality_shard.sh`、`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/compare_generation_outputs.py`、`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/diagnostics/fp8_kv_quality/`。
- 后续注意：正式 B=16 prefill-decode 的 baseline、ours、速度和任务分数均必须用 FP8 KV；旧 BF16-KV B=16 可能形成多个 request wave，不能与该口径混用。需要在这一口径下重测/重标所有 Pareto 点。

## 2026-07-19 - 暂停 FP8 KV cache 路径
- 决策：FP8 KV 已证明能使 B=16/S=2048/O=80 成为单一 prefill wave，但会显著改变 greedy 生成；20 条 IWSLT smoke 未显示明显退化，却不足以证明质量等价或提升。因此暂不将 FP8 KV 纳入正式 runner 和 Pareto 结果。
- 后续方向：优先验证 B=8、BF16 KV。dense-BF16 在 0.80 memory-utilization 下已有 19,840-token capacity，而 B=8 场景仅要求 `8*(2048+80)=17,024` tokens；若 trace 确认为单 wave，便可避免 FP8 近似及 request-wave phase bug。还需实测该 batch 下 prefill/decode 时间占比，以确认阶段异构的研究场景仍有足够代表性。

## 2026-07-19 - B=8 BF16-KV feasibility and phase-balance diagnostic
- 开发目的：验证缩小 prefill-decode batch 是否能在不使用 FP8 KV 和不实现 request-wave reentry 的前提下恢复正确 phase 语义，同时保持 prefill/decode 均有代表性的时延占比。
- 修改内容：benchmark 支持仅用于诊断的 `--batch/--input-seq/--output-seq` 覆盖；CUDA-event profiler 同样参数化。发现并修复 benchmark 顶层启动单样本子进程时未透传这些覆盖参数（以及 `--reuse-llm`）的问题；无效的意外 B=16 诊断已终止且不采信。
- 验证：在 dense-BF16、V1、BF16 KV、`gpu_memory_utilization=0.80` 下，vLLM 可分配 22,672 KV tokens，超过 B=8 所需 17,024。trace 仅有 `prefill/tokens=16384` 128 次和 `decode/tokens=8` 10,112 次，确认单一完整 prefill wave 后才 decode。CUDA Linear 时间为 prefill 892.0 ms（55.0%）与 decode 728.8 ms（45.0%）；同卡 5 次复用测量的 TTFT 为 1021.9 ms、80-token E2E 为 2357.7 ms，E2E 中 prefill 约 43.3%、后续 decode 约 56.7%。
- 影响文件：`artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py`、`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/profile_phase_policy_gpu_time.py`、`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/diagnostics/b8_bf16_kv/`。
- 后续注意：B=8 是可行且 phase-balanced 的候选正式场景；在切换正式实验前，还需在相同 B=8/BF16-KV 口径下重新进行 baseline/ours 的速度校准、Pareto 求解和任务质量评测，不能复用旧 B=16 速度数据。

## 2026-07-19 - B=8/S=2048/O=64 phase-balance validation
- 开发目的：检验更自然的 64-token decoding 长度是否仍保留阶段异构策略所需的 prefill/decode 平衡。
- 验证：dense-BF16、BF16 KV、V1 no-chunk、`gpu_memory_utilization=0.80` 下，trace 为一轮 `prefill/tokens=16384`（128 Linear calls）和一轮 `decode/tokens=8`（8064 calls），无 chunk/wave 问题。CUDA Linear 时间 prefill 892.8 ms（60.6%）、decode 581.3 ms（39.4%）；同卡复用测量 TTFT 1022.4 ms、64-token E2E 2092.4 ms，TTFT/pre-fill 首 token 占 48.9%，后续 decode 占 51.1%。
- 结论：O=64 仍是实测接近 50/50 的 phase-balanced 场景，同时比 O=80 更容易解释；可作为后续正式 B=8 prefill-decode 场景的优先候选。
- 产物：`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/llama2_7b_chat/diagnostics/b8_o64_bf16_kv/`。

## 2026-07-19 - Archive rematerializable legacy phase policies before checkpoint cleanup
- 开发目的：在释放旧 prefill-decode checkpoint 的大体积权重前，保留完整模块级策略定义。
- 修改内容：归档 `035_llama2_prefill_decode_e2e_speed_model/checkpoints` 与 `036_llama2_prefill_decode_intermediate_points/checkpoints` 的所有 `phase_hetero_policy.json` 和 `phase_hetero_manifest.json`，并保留原相对路径、生成 SHA256SUMS 和使用说明。
- 验证：归档 30 个文件（732 KiB）；逐文件 SHA-256 与源文件一致。原 checkpoint weight 目录尚未删除。
- 后续注意：035/036 的 runner 都曾传入 `--prune`，但当前归档策略本身没有 sparse method assignment；归档足以重建模块策略，不能复现旧 runtime 的精确时延。删除由用户执行。
