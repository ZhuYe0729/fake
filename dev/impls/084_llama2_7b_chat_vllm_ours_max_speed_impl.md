# 084 Llama2-7B-Chat vLLM Ours Max-Speed Implementation

## 2026-07-10 - Initial implementation
- 开发目的：为 Llama2-7B-Chat 实现基于 kernel latency predictor 的 vLLM 层异构 max-speed 策略、导出、测速和 PMPD 精度评测流程。
- 修改内容：新增 predictor-only 策略生成器，按 vLLM fused `qkv_proj/o_proj/gate_up_proj/down_proj` 的实际 shape 为 32 层共 128 个模块选择 prefill/decode 后端；将 predictor `marlin_nvfp4` 映射为 runtime `w4a16_ours`，输出候选审计、phase-hetero policy 和元数据。新增既有 phase-hetero exporter 封装、one-shot phase-runtime E2E 测速、PMPD 三数据集评测、汇总和全流程 launcher。
- 影响文件：`artifacts/exports/vllm/ours/llama2-7b-chat/`、`dev/plans/084_llama2_7b_chat_vllm_ours_max_speed_plan.md`、本文件。
- 验证：新增 Python 脚本通过 `py_compile`，launcher 通过 `bash -n`；在 `cospaq` 环境完成两个场景的 predictor smoke，均生成 128-module policy，`prefill_only` metadata 的 `output_tokens=0`，runtime 方法均在支持集合内。
- 后续修正：真实 exporter 对 sparse-NVFP4 要求输入已满足结构化稀疏；launcher 固定传入 `--prune`，使所选 sparse 后端在导出时执行对应剪枝。

## 2026-07-10 - GPU smoke fixes
- 开发目的：在实际 vLLM phase-heterogeneous runtime 上完成 checkpoint 加载与速度 smoke。
- 修改内容：确认 GPU 1 成功导出 prefill-only checkpoint（约 5.1 GiB）；vLLM 能加载 `phase_hetero_mytest` checkpoint。修复 benchmark 在 TTFT one-shot 实例销毁后未释放 CUDA allocator cache、导致 main 实例因显存利用率检查失败的问题；每轮显式 GC、empty cache 和 IPC collect。
- 修正补充：本机 vLLM revision 的 engine 需要显式调用 `llm.llm_engine.engine_core.shutdown()` 才会及时回收 KV cache/model 资源；speed 与 quality 的 one-shot loop 都在释放 CUDA cache 前调用该 API。
- 后续注意：vLLM benchmark/quality 使用本机 `vllm` 环境；`cospaq` 环境缺少 vLLM 所需的 Python 依赖，仅保留为 predictor/export 环境。

## 2026-07-10 - GPU export and vLLM smoke
- 开发目的：验证 predictor policy 能实际导出为 vLLM phase-heterogeneous checkpoint 并被 runtime 加载。
- 修改内容：在空闲 RTX 5090 上完成两个场景的 checkpoint 导出；prefill-only checkpoint 约 5.1 GiB，prefill-decode checkpoint 约 7.3 GiB。测速改为 phase runtime 的独立 one-shot 子进程，避免同一 Python 进程内已释放 phase 权重/KV cache 无法重新 materialize 的限制。
- 验证结果：prefill-only 单次 smoke 完成，E2E 534.259 ms、TTFT 535.068 ms。prefill-decode 单次 smoke 完成，完整 E2E 2978.959 ms；首次 TTFT 包含 CUDA extension JIT 编译（52.141 s），不可用于 TPOT 结论。
- 后续注意：当前 vLLM 环境在新子进程中会重编译部分 sparse extension，导致多样本 phase benchmark 成本过高；正式多次统计前应使 vLLM 使用与现有 CUTLASS extension 二进制匹配的编译环境/缓存，或为 phase runtime 增加单进程 reset/re-materialization API。PMPD 全量质量评测尚未启动，避免在该未解决 runtime 开销口径下长时间占用 GPU。

## 2026-07-10 - Established fresh-process test integration
- 开发目的：对齐 test/vllm 中已验证的 phase-hetero 多次测速和 isolated PMPD 生命周期。
- 修改内容：新增 fresh-process speed launcher，复用 `benchmark_one.py` 的“加载不计时、仅统计 generate”口径，默认每种 output length 1 warmup + 10 measured runs；新增每四样本一个新进程的 PMPD launcher，支持三个 baseline 数据集。原子 prefill-only fresh-process smoke 成功，`generate_s=0.718312s`，初始化 `9.448085s` 单列记录且不进入速度结果。
- 后续注意：全量 PMPD 启动时需要为每个数据集传入精确的 `QUESTION_END`，以保证每批独立进程且不遗漏样本。

## 2026-07-10 - Formal speed and quality launch
- 开发目的：按 test/vllm 已验证的 fresh-process/isolated-process 口径启动正式测试。
- 修改内容：启动两个场景的 speed runner（每种 output length 1 warmup + 10 measured fresh-process runs）；启动 6 个全量 PMPD jobs，分别覆盖两个场景和 CNN/DM 1000、DSum 1500、IWSLT 333。每个 PMPD job 使用四样本一进程、模型初始化不纳入 generation quality 指标。
- 验证：prefill-only fresh-process atomic run 的 `generate_s=0.718312s`；CNN/DM 四样本 isolated PMPD smoke 完成，生成 metrics 和 JSONL，确认 lifecycle 可行。
- 后续注意：全量 quality jobs 仍在后台运行，日志位于 `artifacts/exports/vllm/ours/llama2-7b-chat/max_speed/quality_jobs/`。

## 2026-07-11 - Baseline-aligned prefill-only retest
- 开发目的：消除 fresh-process harness 与正式 baseline 的 workload 配置差异。
- 修改内容：新增 phase one-shot 单次 runner，复用 baseline 的 `TokensPrompt`、固定 greedy output、`max_model_len=2049`、`max_num_seqs=8`、关闭 prefix cache 与“LLM 加载后才开始计时”口径；在 GPU 7 完成 1 warmup + 5 fresh-process measured runs。
- 验证结果：ours max-speed prefill-only median 为 527.083 ms、mean 为 524.574 ms。相对现有 uniform sparse-NVFP4 baseline 520.632 ms 慢约 1.24%，而非之前不同 runner 下看起来的显著差距。

## 2026-07-11 - Baseline-aligned prefill-decode retest
- 开发目的：以和 baseline 相同的 prompt、模型长度、采样及计时口径确认 phase-switch 策略的真实 E2E/TTFT/TPOT。
- 修改内容：在 GPU 7 完成 output=1 和 output=80 各 1 warmup + 5 measured fresh-process runs，按 baseline 公式从两个 median 推导 TPOT。
- 验证结果：ours prefill-decode TTFT median 2066.895 ms、E2E median 3574.655 ms、TPOT 19.086 ms。该策略快于 dense BF16（E2E 1.362x）和 dense NVFP4（1.201x），但略慢于 uniform Marlin NVFP4（E2E 0.978x）及 sparse BF16（0.958x）；TPOT 接近 Marlin（慢约 5.2%）。

## 2026-07-11 - Scenario-specific official speed protocol
- 开发目的：按 workload 语义确定最终汇报的测速口径。
- 修改内容：确定 prefill-only 以 baseline-aligned runner 为正式结果，因为没有 phase transition；prefill-decode 以 test/vllm 既有 fresh-process phase-hetero runner 为正式结果，因为其正确覆盖 prefill-to-decode 权重切换和 one-shot lifecycle。baseline-aligned prefill-decode 数据仅保留为诊断，不能替代 phase-specific reference protocol。
- 正式结果：prefill-only median 527.083 ms；prefill-decode fresh-process 的 TTFT median 1562.697 ms、E2E median 3079.879 ms、TPOT 19.205 ms（均为 10 measured repeats）。

## 2026-07-11 - Eight-GPU full PMPD launch
- 开发目的：补齐两个策略在 CNN/DM 1000、DSum 1500 与 IWSLT 333 上的全量生成质量。
- 修改内容：isolated PMPD runner 支持任意 question range、独立 shard 输出和跳过分片 metrics；新增 8-GPU 动态 shard scheduler（每 shard 至多 360 样本）和 JSONL 合并/最终 metrics 脚本，避免并行 append 同一 JSONL 的竞争。
- 运行状态：18 个 shard 已在 GPU 0–7 启动；每四样本一个 phase-hetero 新进程。预计生成和最终 metrics 合计约 1.5–2 小时。

## 2026-07-11 - Quality scheduler resume fix
- 开发目的：恢复未完成的 IWSLT shard 并执行最终合并。
- 修改内容：修复多线程 scheduler 对共享 GPU list 的竞态，改用线程安全 queue；为完整 shard 增加按期望样本数的 resume 跳过，避免重跑已完成的 CNN/DM 和 DSum shard。
- 当前状态：两个 IWSLT shard 缺失，重启后只应补跑这两个 shard，随后自动合并六组 JSONL 并计算最终 metrics。

## 2026-07-11 - Full PMPD quality completion
- 开发目的：完成六组场景/数据集的合并与质量指标。
- 修改内容：IWSLT 使用 baseline 已采用的 Llama-2-chat tokenizer 作为长度过滤器；补齐两个 IWSLT shard，合并所有 JSONL 并计算 metrics。汇总脚本改为优先读取场景指定的正式 speed protocol，避免旧 smoke CSV 混入最终表。
- 结果摘要：prefill-decode 在 CNN/DM、DSum、IWSLT 上分别为 Rouge-L/BERTScore `23.544/87.082`、`21.581/87.154`、Rouge-L/SacreBLEU `45.309/18.301`，接近 dense baseline。prefill-only 的 mixed sparse 策略质量显著退化，不能作为可用的质量-速度点。

## 2026-07-11 - Unified result summary
- 开发目的：提供两个场景、全部 uniform baseline 与 ours 的单一可读汇总入口。
- 修改内容：新增 `max_speed/summary.md`，汇总 speed、PMPD quality、测速协议、结论和原始 CSV 路径；明确 prefill-only 与 prefill-decode 使用不同的正式 ours speed protocol。
- 后续注意：Pareto 求解和质量建模按计划保留为 TODO。
