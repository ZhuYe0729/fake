# 120 Llama3.1 prefill speed decomposition debug implementation

## 2026-07-20 - Exact local versus E2E decomposition
- 开发目的：区分 Llama3 high-sparse 端的 per-linear latency predictor 误差与 phase-vLLM E2E composition 误差。
- 修改内容：新增 059 独立 debug 目录；使用既有 `KernelLatencyPredictor.profile` 对四个真实 fused shape 和五种方法完成 20 个 exact module-forward measurements；将其与 038 action support、058 closure/anchor E2E 结果逐策略关联。
- 验证结果：p014 local 预测仅高估 5.0%（380.32→362.05 ms），但 E2E 相对 exact local 仍多 377.62 ms（2.043x）。dense-NVFP4 与 sparse-BF16 的 326.37 ms E2E 差中，约 24% 由 exact local 差解释、约 76% 来自 E2E residual。
- 后续注意：058 主结果未被修改。下一步若修复速度模型，应以 high-sparse mixed E2E anchors 拟合 composition-aware calibrator；local 模型仅需补入 Llama3 gate/up sparse-BF16 的 exact row（其原预测低估 15.9%）。

## 2026-07-20 - Static runtime/cache audit (no GPU execution)
- 开发目的：确认 high-sparse 的大幅 E2E 偏差是否可能来自真实 Linear 路径本身，而非先验地归因给非 Linear 固定开销。
- 修改内容：逐行对照 CUTLASS wrapper `forward()` 与 phase-vLLM adapter `apply()`，并检查 sparse/dense backend 的 workspace 与 cache 实现。
- 验证结果：对 prefill `tokens=8*2048=16384`，dense-NVFP4 和 sparse-NVFP4 在两条路径中执行相同的 packing/GEMM 流程；sparse 的 padding 分支也不会触发。更重要的是，`sparse_nvfp4_sm120.cu` 仅维护一个全局 `CachedRunState`，cache key 同时包含权重、activation、scale、output 等指针。单 module 重复 benchmark 可能命中；真实模型逐层的指针会变化，几乎必然连续 miss 并反复 GEMM/workspace 初始化。相比之下 sparse-BF16 的 cuSPARSELt plan/workspace LRU 仅按 shape/device 缓存，可跨同 shape layer 复用；dense-NVFP4 则每次通过 PyTorch allocator 创建临时 workspace。
- 后续注意：将此前的 “E2E residual” 改称为 “standalone local benchmark 未捕获的延迟”。下一步先做 cache-sensitive 的真实 adapter timing，再决定是否调整模型；本次未启动 GPU、未修改 runtime 或主结果。

## 2026-07-20 - Same-runtime `apply()` timing falsifies the large residual
- 开发目的：直接验证 standalone CUTLASS-wrapper local measurement 与 phase-vLLM 实际 linear adapter 是否存在足以解释历史 E2E 残差的偏差。
- 修改内容：新增 synthetic sparse-NVFP4 cache probe，以及 `profile_prefill_vllm_apply.py`；后者在真实 `PhaseHeteroMyTestLinearMethod.apply()` 调用前后记录 CUDA event。以相同导出/同一 vLLM 配置重测 p000（dense BF16）与 p014（high-sparse mixed）。
- 验证结果：synthetic cache probe 的复用/不同输出/逐层循环均为 1.79–1.85 ms/call，没有观察到可解释大残差的 pointer-cache 代价。真实 phase-vLLM `apply` 合计分别为 p000 966.36/981.50 ms、p014 357.46/358.59 ms，分别对齐 standalone exact 968.75/362.05 ms（均在 1.3% 内）；E2E wall 分别为 1089.72/1106.29、479.70/480.69 ms。
- 后续注意：上一条 static audit 中关于 sparse-NVFP4 pointer-key cache “几乎必然导致大 E2E 代价”只是待验证假设，现已被直接 probe 否定。历史 closure 的 p000=1294.20 ms/p014=739.67 ms 与此同 runtime instrumented result 不兼容，应视为旧 runner/测量边界或外部干扰的历史数据，不能用于重拟合 local roofline 模型。下一步应统一并固定 E2E runner 后，再测少量 representative policy 验证 E2E composition；无需重建 local predictor。

## 2026-07-20 - Historical runner compatibility check
- 开发目的：排除 “旧数据仅由外部干扰造成” 的过早结论，并定位同 checkpoint 下历史 benchmark 与新 profiler 的显式配置差异。
- 修改内容：以同一 p000 checkpoint 直接运行历史 `benchmark_phase_baseline_one.py`，得到 1293.11 ms，复现旧 closure 的 1294.20 ms。向 instrumented profiler 增加 `--skip-explicit-prefill-prepare` 开关，准备只改变 `prepare_next_prefill()/wait_for_prefill_ready()` 这一因素的 A/B。
- 后续注意：历史 runner 的数值是可复现的，不是随机 GPU 干扰；当前已知关键差异是它只调用 `enable_phase_hetero()`，而 instrumentation runner 在每个 pass 前显式请求 prefill-ready。下一步以 hook 保持历史 phase setup，记录是否发生 scheduler/phase state 的不同以及各 linear `apply` 的时间。

## 2026-07-20 - Scheduler-cap A/B preparation
- 开发目的：隔离历史 benchmark 与 instrumentation runner 的另一个实质配置差异：前者未固定 `max_num_batched_tokens`，后者设为完整 prefill workload 的 16384。
- 修改内容：新增 `benchmark_phase_controlled.py`，保留历史 fresh-process `generate_only_after_loaded_llm` 的计时边界，但强制指定 scheduler cap；不添加 per-linear CUDA event hook。
- 后续注意：该脚本用于验证完整 B=8×2048 prefill 是否因默认 cap 被拆成多波；若 16384 cap 与约 1090 ms profiler E2E 对齐，则历史 1294/740 ms 不能再与 per-linear predictor 直接比较，正式 protocol 必须固定该 cap。

## 2026-07-20 - Scheduler-cap and cold-request diagnosis closed
- 开发目的：完成 p000/p014 两端的受控 A/B，验证历史 Llama3 prefill E2E 偏差来源。
- 修改内容：在同 checkpoint、同 fresh-process `generate` 边界下，使用新 controlled runner 固定 `max_num_batched_tokens=16384`；并与已采集的 warm phase-vLLM `apply` profile 对照。
- 验证结果：p000：历史 1294.20 ms，cold cap=16384 为 1149.07 ms，warmed cap=16384 为 1089.72/1106.29 ms；p014：历史 739.67 ms，cold cap=16384 为 529.16 ms，warmed 为 479.70/480.69 ms。两端的真实 `apply` sum 仍紧贴 standalone exact local（p000 966–982 vs 968.75；p014 357–359 vs 362.05）。
- 后续注意：历史 runner 每个所谓 repeat 都是一个 fresh process 的首请求，且默认 scheduler cap 未固定，因而把 cold plan/setup 与 policy-dependent prefill-wave scheduling 混入 E2E。正式 Llama3 prefill protocol 应固定 B=8、L=2048、`max_num_seqs=8`、`max_num_batched_tokens=16384`、eager/无 chunked prefill/无 prefix cache，在同一 loaded engine 做一次不计时 warmup 后连续五次计时；随后重测 speed anchors/closure 并重拟合 E2E calibrator。local roofline predictor、canonical states、quality proxy 和 solver 无需重建。

## 2026-07-20 - GPU-free revalidation preparation
- 开发目的：在 GPU 忙碌期间固化这次测速 bug 的规避方式，并为后续 Llama3 prefill E2E 重测做好可恢复代码准备。
- 修改内容：创建 123 号计划与隔离的 `061_llama31_prefill_warmed_speed_revalidation`；新增固定协议 benchmark、058 12-anchor design 复用、每卡一个 policy 的调度器、以及不读取历史 speed label 的 monotone calibrator。benchmark JSON 会保存 warmup、五次 timed samples、scheduler cap 与 protocol。
- 后续注意：061 尚未启动任何 GPU 测试；待 GPU 空闲后先运行 `build_speed_design.py`、`run_speed_anchors.py`、`fit_speed_calibrator.py`，然后才用新 `calibration.csv` 重解与 closure。058 保持只读。

## 2026-07-20 - Protocol guardrail documentation
- 开发目的：使后续执行者无需回溯本次 debug 过程，也不会把历史 cold/fixed-default-cap 的速度标签混入新的校准数据。
- 修改内容：新增 061 `SPEED_PROTOCOL.md`，记录根因、强制 runtime 参数、pre-measurement checklist 与 JSON 验收字段。
- 后续注意：只有带有 `same_engine_warmup_then_timed_explicit_prefill_phase` 协议标识且含五次 `timed_ms` 的结果，才可作为本轮 calibrator/closure speed label。

## 2026-07-20 - Full post-calibration path prepared
- 开发目的：使 GPU 空闲后可从 anchor 测速一路执行到新的策略求解与 closure speed，而不需要临时改脚本。
- 修改内容：061 在 build 阶段只复制已拟合的 058 quality model；新增 solver wrapper（复用未改动的离散约束优化公式，但从 061 读取新 calibration）和 generic solved-policy closure speed runner。
- 后续注意：061 的 solver 仍严格复用 058 canonical local error/features 与 quality model；唯一更换的优化输入是 corrected warmed E2E speed calibration。

## 2026-07-20 - Warmed-runner phase initialization correction
- 开发目的：首次实际执行 061 anchors 时发现并修复 phase runtime 的合法状态转换。
- 修改内容：首次 warmup 不再错误调用 `prepare_next_prefill()`（runtime 初始已是 prefill，且该 API 需要前一次 decode 已完成）；后续五次 timed request 及额外 warmup 才显式切回 prefill。
- 后续注意：首批任务在输出 timing JSON 前失败，未产生可用标签；重启时必须从零重测。该约束已写入 protocol 文档。

## 2026-07-20 - Serialized-export throughput guardrail
- 开发目的：首轮 5 卡并行时，多个 8B exporter 在磁盘/初始化阶段全部停滞（临时目录各仅约 11 MB），造成 GPU 空闲。
- 修改内容：`run_speed_policy.py` 通过 experiment-local `export.lock` 串行化 checkpoint export；锁在 export 完成后释放，因此一张卡可 benchmark 已导出 checkpoint 的同时，另一 worker 开始下一个 export。
- 后续注意：这是 I/O/初始化并发保护，不会串行 GPU benchmark。已完成且协议合格的 p00/p02/p04 保留；停滞任务将在重启后从缺失 JSON 恢复。

## 2026-07-20 - Lazy canonical-state export correction
- 开发目的：避免 dense-only anchor 无谓加载两套 large canonical sparse state（p01 为 128 个 dense-NVFP4，却在旧 wrapper 中仍加载 sparse BF16/NVFP4 state）。
- 修改内容：061 runner 直接调用 phase exporter，解析 policy 后仅为实际出现的 `sparse_bf16`/`sparse_nvfp4` 传入相应 canonical state；dense-only policy 不再经由无条件传入两 state 的 057 wrapper。
- 后续注意：这只减少导出开销，不改变任何 sparse policy 的权重来源或 runtime；重启后的缺失 policy 将使用该路径，现有合规 p00/p02/p04 继续复用。

## 2026-07-20 - Dense-NVFP4 export stall diagnostic
- 开发目的：061 的 dense-NVFP4 p01 export 在 GPU 几乎无计算的状态下停留约 30 分钟，需先区分底层 pack kernel stall 与 checkpoint exporter orchestration 问题。
- 修改内容：新增单个 4096×4096 BF16 权重的 `quantize_weight_bf16` GPU probe；不读取模型或 canonical state。
- 后续注意：该 probe 只用于诊断。若成功快速完成，问题在 full exporter/权重搬运路径；若同样停滞，需修复或预编译 cutlass wrapper 后再重启 anchors。

## 2026-07-20 - Reuse verified dense-NVFP4 packs
- 开发目的：minimal probe 在 cospaq/vLLM 两环境均停滞，确定当前 online `quantize_weight_bf16` native path 不可用于生成 anchors。
- 修改内容：新增 export wrapper，将 external phase exporter 的 dense-NVFP4 quantizer monkey-patch 为读取已有 uniform dense-NVFP4 checkpoint 的 `weight/weight_scale/weight_global_scale`。它们是同一方法、同一模型的真实预打包结果；sparse 模块仍严格使用对应 canonical SparseGPT state。
- 后续注意：此改动避免在线 pack，不改变 dense-NVFP4 的权重值或 phase runtime；重新启动前先对 uniform p01 做导出/运行 smoke test，再恢复剩余 anchors。

## 2026-07-20 - vLLM-specific dense-NVFP4 extension cache
- 开发目的：shared extension cache 下 pack/runtime 均停滞；需要验证是否为 cospaq/vLLM runtime ABI/cache collision。
- 修改内容：使用 `CUTLASS_WRAPPER_NVFP4_EXT_BUILD_DIR=061/vllm_nvfp4_extension` 在 vLLM 环境重建 extension。最小 4096×4096 pack 成功完成（约 119.8 s，包含首次 build/init）；061 runner 现强制为 vLLM benchmark 使用该独立目录。
- 后续注意：dense-NVFP4 pack 不应再在 cospaq exporter 中触发（仍复用 uniform packed state）；后续 p01 phase runtime smoke test 必须继承此专属 cache，验证通过后再恢复 anchors。

## 2026-07-20 - Full extension-cache namespace separation
- 开发目的：p03 canonical sparse-NVFP4 export 同样停滞，源码确认 sparse-NVFP4 默认也使用 shared `torch_extensions`，存在同一 ABI/cache collision 风险。
- 修改内容：061 runner 现区分 exporter 的 cospaq sparse-NVFP4/BF16 extension 目录和 vLLM benchmark 的 sparse-NVFP4/BF16 extension 目录；dense benchmark 继续使用已验证的 vLLM directory。
- 后续注意：当前已经启动的 p03 使用旧环境，不能通过修改中的新变量修复；取得终止授权后需从 p03 起重启缺失 anchor，新的任务将触发各自环境的一次独立 build。

## 2026-07-21 - Restart corrected warmed anchor calibration
- 开发目的：在取得终止授权后清理旧 cache 配置启动的卡死 exporter，并从已有合规 JSON 继续补全 061 的测速锚点。
- 修改内容：终止旧 scheduler、worker、exporter 及 orphan probe；确认 GPU 1 有外部 26 GiB 占用后，改用 GPU 4、3、5、6。closure speed runner 同步改为复用预打包 dense-NVFP4、按策略懒加载 canonical sparse state，并使用与 anchor 一致的环境隔离扩展缓存和 export lock。
- 影响文件：`artifacts/debug/061_llama31_prefill_warmed_speed_revalidation/scripts/run_closure_speed_policy.py`。
- 后续注意：p00/p02/p04 已有五次合规样本；p01 的单样本 smoke JSON 仅作诊断，不能进入 calibrator。待重启 worker 产出 p01.json 后再拟合。

## 2026-07-21 - Warmed speed, NLL, and downstream-task closure
- 开发目的：完成 061 的重测速后约束求解，并以真实 phase-vLLM NLL 与下游任务验证新策略。
- 修改内容：12 个 speed anchor 全部采用 fixed-cap、同 engine warmup + 5 次测量；重拟合 monotone E2E calibration（独立 holdout MAE 17.85 ms），重解 15 个策略并对 8 个代表/max-speed 点完成速度、100-block WikiText NLL、以及 WikiText/Winogrande/ARC-Easy/ARC-Challenge/MMLU。新增 061 closure NLL/task runner，导出阶段复用 prepacked dense-NVFP4/canonical sparse state 与环境隔离 extension cache；生成 measured task CSV、Markdown 总表和五张 task Pareto 图。
- 影响文件：`061/scripts/run_closure_nll_policy.py`、`run_closure_nll_selection.py`、`run_task_policy.py`、`run_task_selection.py`、`build_task_pareto_report.py`，以及 `061/llama31_8b_instruct/report/`。
- 验证结果：速度从 1087.53 ms（BF16）到 481.16 ms（2.26x max-speed）；实测 ΔNLL 随速度单调增大；所有 8×5 下游任务结果完整。`point_009` 以 1.91x 接近 sparse-NVFP4 的 1.93x，但 ARC-Challenge 0.4923 显著高于其 0.2637。
- 后续注意：uniform task score 目前引用冻结的 058 实测结果；061 只新增求解 mixed points。论文图必须说明 uniform scores 的来源，且不要将 061 speed calibration 预测值作为实测速率。

## 2026-07-21 - Consolidate corrected Llama3 prefill artifacts
- 开发目的：将 061 新口径结果接入两模型两场景的统一汇总，并创建不覆盖原表的论文表格 v2。
- 修改内容：新增 `060/.../llama31_8b_instruct/prefill_only_02/`，保存新完整表、speed calibration、solver 点、8 个闭环策略和五张 task Pareto 图；复制 `result.tex` 为 `result_v2.tex`，仅更新 RTX 5090 Llama3 prefill-only 各 uniform TTFT 与 ours 的 balanced/max-speed 行。
- 影响文件：`artifacts/debug/060_two_model_two_scenario_result_consolidation/llama31_8b_instruct/prefill_only_02/`、`result_v2.tex`。
- 后续注意：result_v2 的 Llama3 Balanced 选 `point_009`（1.91x），Max speed 选 `point_014`（2.26x）；prefill-decode 列保持原值，原 `result.tex` 未修改。

## 2026-07-21 - Correct Llama2 prefill-decode uniform consolidation
- 开发目的：修复 060 中 Llama2 prefill-decode uniform 下游任务行缺失/混淆，确保不把其他模型的分数归入 Llama2。
- 修改内容：从 056 的 `speed/uniform_baselines.csv` 和 `task_quality/report/task_pareto_points.csv` 导入明确的 uniform 表与原始 CSV；p00--p02 填入已测的 CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU，p03/p04 仅保留速度并显式标示任务未测。复制包含 uniform 标记的 056 Pareto 图。
- 影响文件：`060/.../llama2_7b_chat/prefill_decode/summary.md`、`data/uniform_baseline_results.csv`、`data/uniform_baselines.csv`、`data/task_pareto_points_from_056.csv`、新增 `*_with_uniform_056.png`。
- 后续注意：056 的 sparse-NVFP4 是 prefill sparse / decode dense 的 legal projection，不能简称为全阶段 sparse-NVFP4；不要用任何旧 060 末尾 p00--p04 的隐式行补齐 p03/p04 任务分数。

## 2026-07-21 - Restore Llama2 p03/p04 task-runner import path
- 开发目的：补测 Llama2 prefill-decode uniform p03/p04 的主下游指标时，任务在 phase-vLLM 首个 linear 调用处报 `ModuleNotFoundError: cutlass_wrapper`。
- 修改内容：将本地补测 runner 恢复为历史成功脚本的环境：`REPO=/home/agent/wja/project/my/cospaq/fake`，且将 wrapper package root（`fake/kernels/cutlass/cutlass_wrapper`）直接放入 `PYTHONPATH`；删除误设的上层目录路径。vLLM Python 下最小 `import cutlass_wrapper` 已验证成功，p03 smoke shard 已越过旧的 import 失败点并正常进入 `phase_hetero_mytest` 模型加载。
- 影响文件：`artifacts/debug/056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/run_uniform_task_quality_shard.sh`。
- 后续注意：待首个 p03 runtime 初始化完成并开始写 JSONL 后，再并行补齐 p03/p04 的 CNN/DM、DialogSum、IWSLT 主指标；不要改动已有速度结果。

## 2026-07-21 - Reuse p03 validated extension caches for uniform task backfill
- 开发目的：修复 import 后，p03 仍在 vLLM profile run 中停滞；同入口的 p04 一条样本在约 7 秒内完成，故问题限定为 p03 sparse-NVFP4 runtime 的默认 extension cache。
- 修改内容：p03 runner 自动指定 056 speed 流程已构建的 `vllm_sparse_nvfp4_p03` 与 `vllm_nvfp4_p03` cache。缓存复用后 p03 1 条 smoke 的 engine profile 约 2 秒完成、生成耗时 9.56 秒，并写出 JSONL。新增仅覆盖 p03/p04 三个主指标的分片队列脚本，使用 GPU 0、2--7 串行队列并行补测。
- 影响文件：`run_uniform_task_quality_shard.sh`、`launch_uniform_p03_p04_backfill.sh`、`start_uniform_p03_p04_backfill.sh`。
- 后续注意：这些脚本只补下游任务质量；不修改、也不重测 056 已固定的速度数据。完成后须从 24 个 shard 合并 JSONL，计算 CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU，并更新 060 汇总。

## 2026-07-21 - Sequential fresh-engine memory-release guard
- 开发目的：初次并发补测时，部分 GPU 的首个 IWSLT shard 成功，但紧接的下一个 fresh vLLM engine 因前一进程/NCCL 显存尚未完全释放而被拒绝启动（可用 12.23 GiB，小于 0.75 utilization 所需 23.53 GiB）。
- 修改内容：每个同 GPU 串行队列中两次 shard 之间等待 20 秒，仅作用于进程退场和显存释放，不改变输入、batch、KV cache、phase runtime 或模型策略。
- 影响文件：`launch_uniform_p03_p04_backfill.sh`。
- 后续注意：已退出队列需要仅重启其失败/未启动 shard；已有 JSONL 不得覆盖或重跑。

## 2026-07-21 - Complete uniform p03/p04 primary task backfill
- 开发目的：补齐 060 汇总中 Llama2 prefill-decode uniform sparse-NVFP4 legal projection（p03）和 W4A16（p04）缺失的三项主任务分数。
- 修改内容：24 个 PMPD shard 均完成；新增 primary-only merge，严格校验 CNN/DM=1000、DialogSum=1500、IWSLT=333 样本，复用 PMPD 的 RougeScorer（stemmed ROUGE-L）和 SacreBLEU 公式，但不为当前仅需主指标的回填触发 BERTScore。p03: 15.96/13.57/1.57，p04: 23.73/21.77/18.85（CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT SacreBLEU）。更新 060 的 CSV 与 Markdown 表。
- 影响文件：`merge_uniform_p03_p04_backfill.py`、`task_quality/report/uniform_p03_p04_backfill_summary.csv`、`060/.../prefill_decode/{summary.md,data/uniform_baseline_results.csv}`。
- 后续注意：p03 的 decode 是 dense-NVFP4 legal projection；p03/p04 的 BERTScore 未补测，不能填写或从其他实验推断。

## 2026-07-21 - Fill Llama2 paper-table uniform task cells
- 开发目的：将已汇总的 Llama2 prefill-decode uniform 主指标填入论文表 `result_v2.tex`。
- 修改内容：补齐 BF16、dense-NVFP4、sparse-BF16、p03 legal sparse-NVFP4 projection、W4A16 的 CNN/DM ROUGE-L、DialogSum ROUGE-L、IWSLT BLEU；同时修正已知的 ours max-speed IWSLT BLEU 误填（15.57 改为 b8o64009 实测值 2.60）。
- 影响文件：`artifacts/debug/060_two_model_two_scenario_result_consolidation/result_v2.tex`。
- 后续注意：表中的 Sparse NVFP4 prefill-decode 行必须在论文正文/脚注说明为 decode dense-NVFP4 的 legal projection。

## 2026-07-21 - Adopt ARC-Easy `acc` for prefill-only reporting
- 开发目的：使 prefill-only 下游任务口径与指定 benchmark mapping 一致：ARC-Easy 使用 `acc`，ARC-Challenge 保持 `acc_norm`。
- 修改内容：Llama2/Llama3 报告脚本均从保留的 lm-eval `result.json` 重提取 `arc_easy` 的 `acc,none`；重绘 ARC-Easy Pareto 图。新增 consolidation rebuild 脚本，同步 054 Llama2 和 061 warmed Llama3 的 CSV、Markdown、图表到 060；`result_v2.tex` 的两模型全部 ARC-E 数值同步更新。
- 影响文件：`054/.../build_pareto_validation_report.py`、`061/.../build_task_pareto_report.py`、`060/.../scripts/rebuild_prefill_arc_easy_acc.py`、060 prefill-only bundles、`result_v2.tex`。
- 后续注意：论文中应明确写 ARC-Easy `acc`、ARC-Challenge `acc_norm`；不要再将 ARC-E 列称为 normalized accuracy。

## 2026-07-21 - Diagnose Llama3.1 legacy versus native chat-template generation quality
- 开发目的：在不改动任何主实验结果的前提下，判断 Llama3.1-8B-Instruct 在 prefill-decode 生成任务中的低绝对分数是否主要来自 legacy/common PMPD prompt 与其官方 chat template 不匹配。
- 修改内容：新增隔离目录 `artifacts/debug/062_llama31_prompt_template_diagnosis/`。在 dense-BF16、同一 vLLM backend、相同的固定 PMPD 子集（CNN/DM、DialogSum、IWSLT 各 100）、greedy/256-token 上限下，对比 legacy Human/Assistant prompt 与 `apply_chat_template(..., add_generation_prompt=True)`；统一使用 Llama3 的 EOS/EOT token id 128009。报告校验两臂 question ID/reference 完全一致，记录主指标、生成长度、finish reason、正文角色续写和配对样例。native 相比 legacy 的 CNN/DM ROUGE-L 为 20.56 vs 15.66、DialogSum ROUGE-L 为 17.80 vs 12.02、IWSLT BLEU 为 27.44 vs 9.88；legacy 三任务分别有 99/97/97 条撞到长度上限，native 均为 100 条正常 stop。
- 影响文件：`dev/plans/124_llama31_prompt_template_diagnosis_plan.md`、`artifacts/debug/062_llama31_prompt_template_diagnosis/{scripts,outputs,report}/`。
- 后续注意：native template 结果仅用于解释跨模型绝对分数，不能替换 legacy/common 主表；同一 Llama3 内 BF16/uniform/ours 均固定 legacy 协议，因此其相对压缩比较仍有效。
