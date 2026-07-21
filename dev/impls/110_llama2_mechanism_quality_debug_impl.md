## 2026-07-17 - Initialize mechanism-quality debug bundle
- 开发目的：将量化、稀疏和二者交互拆分为自然可解释的 real-vLLM NLL 代理特征。
- 修改内容：建立 047 debug 目录及其独立校准、拟合和重求解流程；不修改 046 主结果。
- 后续注意：新增策略必须先完成 fixed-block real-vLLM NLL 后才可用于模型选择或重求解。

## 2026-07-17 - Mechanism calibration and v2 proxy implementation
- 开发目的：以自然的机制分解测试并修正稀疏与 NVFP4 混用时的质量低估。
- 修改内容：生成 18 个 hash 固定的 quant-only、sparse-only、co-located 和 separated 校准策略；实现可重启的真实 vLLM NLL runner、NLL 合并与 provenance 校验、零 BF16 偏置的非负 Q/S/S²/S×Q 代理模型、诊断报告及 v2 约束求解器。
- 影响文件：`artifacts/debug/047_llama2_prefill_mechanism_quality_debug/` 下的 `policies/`、`manifest.json` 和 `scripts/`。
- 验证：所有 Python 脚本已通过 `py_compile`；18 项真实 NLL 已在 GPU 1–6 启动，首两项完成且无失败。
- 后续注意：只在 holdout 误差报告确认改善后，才使用 v2 重求解并对新策略进行实测。

## 2026-07-17 - Complete calibration and correct the zero anchor
- 开发目的：验证 v2 能否在真实 vLLM 标签上同时刻画低敏感量化和稀疏累积。
- 修改内容：完成 18/18 个 100×2048 fixed-block NLL 标签并通过 policy/sample hash 校验；对高量化点 `q120` 以较低 KV-cache 预留重试，避免 prompt-logprob 临时 softmax OOM，未改变评分语义。发现 `softplus(0)>0` 使“非负”模型没有零误差锚点，改为 ReLU 非负系数并使用较弱 L2。
- 验证：旧 holdout 的 v2 MAE/RMSE 从 v1 的 0.1214/0.1493 改善至 0.0849/0.1113，Spearman 从 0.8204 改善至 0.8741；但机制 holdout MAE 仍为 0.2145，不能作为最终求解模型。
- 后续注意：下一轮需在不改变 Q/S 机制分解的前提下，引入能表达“低敏感模块选择”的自然策略分布特征或补充该分布的校准点；当前不运行 v2 重求解。

## 2026-07-17 - Document temporary quality-model design
- 开发目的：使当前的特征、公式、NLL 口径、数据划分、已验证的改善与未解决的外推问题可审阅。
- 修改内容：在 `047` 中新增 `TEMP_QUALITY_MODEL.md`，明确 v2 仅为调试模型，且不把下游任务标签或速度数据混入精度拟合。
- 后续注意：后续任何 v3 特征均应以该文档所列 6 个 mechanism holdout 为门槛，而不能只查看旧 holdout。

## 2026-07-17 - Start sensitivity-coverage calibration
- 开发目的：先单独验证增加符合优化器选择分布的训练数据，是否足以修正 v2 对低敏感策略的高估。
- 修改内容：新建隔离的 `048_llama2_prefill_quality_coverage`，生成 24 个训练专用策略：量化、稀疏和混合策略分别覆盖中/高局部敏感度，同时不修改 `046` 或 `047` 的所有 holdout。
- 验证：策略生成器和 runner 已通过 `py_compile`；24 个 fixed-block real-vLLM NLL 任务已在 GPU 1–6 启动。
- 后续注意：完成后只重拟合现有 Q/S 公式并比较 frozen holdout；若仍不够，再独立测试新的策略分布特征。

## 2026-07-17 - Complete sensitivity-coverage ablation
- 开发目的：验证仅增加中/高局部 MSE 选择策略的数据覆盖，能否改善现有 Q/S 公式。
- 修改内容：完成 24/24 条真实 vLLM NLL 标签、provenance 合并，以及保持公式完全不变的重拟合；将结果写入 `048/report/SUMMARY.md`。
- 验证：旧 holdout MAE/RMSE 从 0.0849/0.1113 变为 0.0879/0.1286；mechanism holdout 从 0.2145/0.2273 变为 0.2677/0.3080，均退化。
- 后续注意：不可再以同一 local-MSE 排序派生更多训练策略；须先替换或校准局部精度特征，再判断是否需要补充标签。

## 2026-07-17 - Fisher-weighted local-feature ablation
- 开发目的：检验用 dense next-token NLL 的模块输出 Fisher 敏感度，为局部压缩 MSE 注入下游传播重要性，能否修正 feature 的尺度问题。
- 修改内容：在独立 `049` 中采集 224 个线性子模块的 8×256 WikiText 输出均方梯度；比较原 MSE、MSE×sqrt(Fisher)、MSE×Fisher，拟合公式与冻结划分均保持不变。
- 验证：mechanism holdout MAE 分别为 0.2145、0.2201、0.2231，Fisher 变体未改善；不进入求解流程。
- 后续注意：仅靠 dense 模型的一阶局部传播权重不足，需要考虑与真实运行时压缩算子一致的策略级校准或重新审视 NLL 标签的可分辨尺度。

## 2026-07-17 - Original-formula ReLU control
- 开发目的：按要求隔离检验“回到 046 原方法，仅将 softplus 换为 ReLU”的效果。
- 修改内容：在独立 `050` 中保留原始 feature aggregation、global/method/bucket/type 加性系数、bias、训练步数与正则；仅使用 ReLU 系数和避免零梯度的微小正初始化。
- 验证：原公式 ReLU 的 holdout MAE/RMSE 为 0.1931/0.2642，显著差于原 softplus v1 的 0.1214/0.1493。
- 后续注意：不能把 `047` 的改善简单归因于 ReLU；ReLU 需与零截距、逐模块 Q/S 表达一起评估，当前也不可将 `050` 用于求解。

## 2026-07-17 - Start q120 critical-module ablation
- 开发目的：直接验证 uniform dense-NVFP4 的 NLL 是否由 q120 保留的 8 个模块集中主导。
- 修改内容：新建隔离 `051`，从近无损 q120 出发，生成 8 个 leave-one-protected-module 策略；每个策略仅新增一个 dense-NVFP4 模块。
- 验证：策略 JSON 已生成并通过 Python 静态检查；8 个相同 fixed-block real-vLLM NLL 任务已在 GPU 1–7 启动。
- 后续注意：以 q120 为共同基线报告边际 ΔNLL；单模块不足以解释 uniform 损失时才测小规模组合，不直接将结果拟合进新模型。

## 2026-07-17 - q120 leave-one results and backend control
- 开发目的：解释 q120 与 uniform dense-NVFP4 的 NLL 差异。
- 修改内容：8/8 leave-one 结果完成；每个单模块边际变化均在约 ±5e-5。发现原 uniform `p01` 与 phase-hetero q120 使用不同的量化接口，故其 0.0538 NLL 差不能归因于 8 个模块。
- 后续注意：已启动 `q128_phase` 公平接口对照；只有该点与 q120 的差才可用于判断 8 模块集合的交互效应。

## 2026-07-17 - Complete phase-interface control
- 开发目的：将模块交互假设与 uniform/phase 后端差异区分。
- 修改内容：完成 `q128_phase`，其 128 个 prefill 模块均为 dense-NVFP4 且 trace 确认 `apply_prefill: 128`；汇总写入 `051/SUMMARY.md`。
- 验证：q128_phase ΔNLL=0.000392，q120=0.000304，差仅 0.0000876；uniform p01 ΔNLL=0.053822，且使用 `nvfp4_mytest` 而非 `phase_hetero_mytest`。
- 后续注意：在验证两条 NVFP4 路径具有等价量化/激活语义前，不能再用 uniform p01 与 phase-heterogeneous 点的 NLL 比较来评判精度模型或帕累托支配关系。

## 2026-07-17 - Forensic separation of checkpoint and runtime effects
- 开发目的：量化 historical uniform dense-NVFP4 与全 NVFP4 phase 点之间质量差异来自何处。
- 修改内容：在 `052` 新增直接从原始 BF16 导出 uniform 格式的受控 exporter，并完成 20-block NLL、packed tensor 逐字节比较与总结文档。
- 验证：direct-uniform 与 phase-q128 的 128 个 packed weights 和 256 个 scales 完全一致；但 NLL 分别为 2.042132 / 2.001808（BF16=1.999849）。历史 prepared uniform 为 2.049706；prepared 再打包额外带来 0.007574 NLL，其余 0.040324 来自 `nvfp4_mytest` 与 `phase_hetero_mytest` runtime 语义。
- 后续注意：统一方法与 ours 的 prefill-only 精度曲线目前不可直接比较；必须统一 runtime/activation 语义后才可作论文级 Pareto 支配结论。

## 2026-07-17 - Phase runtime throughput control
- 开发目的：确认将 uniform 策略统一到 phase runtime 是否会引入显著速度代价。
- 修改内容：在 `052` 导出 prefill/decode 均为 dense-NVFP4 的 128-module phase-degenerate uniform checkpoint，并按 B=8、S=2048、O=1、warmup+5 fresh-process 协议对比直接 uniform checkpoint。
- 验证：`phase_hetero_mytest` / `nvfp4_mytest` 的五次中位数为 656.298 / 662.638 ms，phase 快 0.96%，小于重复离散度；dispatcher 没有可观测速损。仅 prefill NVFP4、decode BF16 的双权重 checkpoint 则为 1068.951 ms 且在 0.9 KV 预留 OOM，不能作为 uniform control。
- 后续注意：主比较可将 uniform 写为 phase-degenerate policy；prefill-decode 仍应以同样方式单独验证其阶段组合与显存占用。

## 2026-07-17 - Phase prefill-decode throughput control
- 开发目的：验证 phase runtime 在真实 prefill/decode 阶段切换工作负载中是否有额外 E2E 开销。
- 修改内容：复用相同 original-weight、all-dense-NVFP4 对照 checkpoint，在 B=16、S=2048、O=80 下各完成 warmup+5 fresh-process 运行。
- 验证：phase / uniform 的五次 E2E 中位数为 4400.770 / 4490.281 ms，差 2.0%，处于跨 GPU 重复波动内；没有可测得的 phase dispatch 速度损失。
- 后续注意：两场景都可以以 phase-degenerate uniform policy 作为主精度/速度比较口径；不将小幅正差异表述为 phase 加速。

## 2026-07-17 - Phase-unified Llama2 quality recalibration
- 开发目的：在修复 legacy uniform 重复 pack 与 runtime 不一致后，检验原 046 精度代理公式是否自然改善。
- 修改内容：新建隔离 `053_llama2_prefill_phase_unified_quality_recalibration`，复制并 hash 校验 046 的 72 条策略和固定 WikiText blocks；全部压缩策略以 phase-degenerate export 重测，保持特征、softplus local+global 公式、3000 steps 与 54/18 划分不变。
- 验证：72/72 个 100×2048 NLL 标签通过 sample/policy hash、token count 与 phase backend 校验；仅 `p01`--`p04` 标签改变，p01 ΔNLL 从 0.053822 降至 0.042103，p02/p03 从 0.345707/1.147809 升至 1.801113/6.120044。新模型 holdout MAE/RMSE/Spearman 为 0.425113/0.545739/0.6945，未优于 046 的 0.121419/0.149305/0.8204。
- 后续注意：修正标签揭示原公式无法同时拟合 phase-unified uniform sparse 端点和 mixed-policy 标签；不能将 046 的较好误差继续当作公平口径下的精度模型表现，下一轮需在不改变评测口径前提下重新设计自然的 sparse/quantization 特征或训练策略覆盖。
