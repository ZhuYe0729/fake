# 115 Llama2 canonical prefill-decode Pareto

## 2026-07-17 - Isolated bootstrap and runtime smoke gate
- 开发目的：先证明 prefill/decode 双阶段的 canonical sparse 导出、真实 vLLM teacher forcing 和阶段切换可用，再开始大规模校准。
- 修改内容：创建 debug 055 输入 bundle；复用 054 的 immutable canonical sparse BF16/NVFP4 states；增强 decode-NLL streamer 支持 canonical state、禁止该模式下 direct prune，并持久化 exporter provenance。用含两种 sparse method 的双阶段 policy 完成 phase trace smoke。
- 影响文件：`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/`、`artifacts/debug/044_llama_prefill_decode_vllm_nll/scripts/stream_phase_policy_nll.py`。
- 验证：`smoke/result_provenance.json` 中 `enter_decode=1`、`apply_decode=10112`；对应 export provenance 记录两套 canonical state 且 `prune=false`。
- 后续注意：033 的旧 policies 可作为 calibration 设计来源，但其 direct-prune NLL 不可作为 label；所有 NLL 必须重新使用 055 canonical 流程测得。

## 2026-07-17 - Canonical quality-label pilot launched
- 开发目的：在投入 72-policy 重标注前，先验证 uniform endpoints 的真实 phase-switch NLL 与 canonical sparse source 行为。
- 修改内容：将 033 的 72 个 controlled prefill-decode policy 设计及其 54/18 split 隔离复制到 055；并行启动 p00--p04 的 16-block canonical teacher-forced NLL pilot。
- 影响文件：`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/stage_calibration_policies.py`、`llama2_7b_chat/policies/prefill_decode/`、`llama2_7b_chat/nll/pilot/`。
- 后续注意：pilot 只用于验证；完成后以同一 evaluator 重新采集完整 calibration labels，不能回填旧 033 NLL。

## 2026-07-17 - Pilot passed; full-label dispatcher prepared
- 开发目的：验证 endpoint 的 canonical NLL 尺度并准备完整重标注。
- 修改内容：p00--p04 的 16-block pilot 全部完成，relative ΔNLL 为 dense-NVFP4 `0.044848`、sparse-BF16 `0.417955`、sparse-NVFP4 `0.339666`、W4 `0.021239`；新增 72-policy、GPU 多 worker、可恢复 NLL dispatcher。
- 影响文件：`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/run_canonical_nll.py`、`llama2_7b_chat/nll/pilot/`。
- 后续注意：完整 100-block dispatcher 的 GPU 启动受平台 Codex 用量限制拒绝；在权限恢复前不可绕过或改用间接启动。

## 2026-07-18 - 修复 full-label GPU 调度并恢复补测
- 开发目的：消除 full 100-block canonical NLL 采集中的非算法性 OOM，保留已完成的有效标签。
- 修改内容：将 dispatcher 改为每张 GPU 一个串行 worker（不再按 future 序号轮转 GPU）；仅清理缺少最终 JSON 的策略残留临时 checkpoint，并仅重跑缺失项。
- 影响文件：`artifacts/debug/055_llama2_prefill_decode_canonical_pareto/scripts/run_canonical_nll.py`、`logs/nll_full_dispatcher_retry.log`。
- 验证：恢复前已有 54/72 个有效 JSON；补测启动后 GPU 1--7 各仅有一个 `stream_phase_policy_nll.py`，避免同卡同时导出模型。
- 后续注意：必须待 72 个 JSON 均完成后再合并标签和拟合阶段质量模型；不能把 OOM 失败日志当作质量数据。

## 2026-07-18 - 完成 canonical phase-switch NLL 标签并开始局部特征采集
- 开发目的：为联合 prefill/decode 质量模型提供同一真实 vLLM 口径的端到端标签与 canonical sparse 局部误差。
- 修改内容：72/72 个 100-block 标签合并为 `nll/prefill_decode.csv`，并逐项验证 `enter_decode` 与 `apply_decode` trace；新增 canonical phase-local error collector，分别测量 prefill/decode 下 sparse BF16/NVFP4 wrapper 相对 BF16 模块输出的误差。
- 影响文件：`llama2_7b_chat/nll/prefill_decode.csv`、`scripts/merge_canonical_nll.py`、`scripts/collect_canonical_phase_local_errors.py`、`llama2_7b_chat/local_errors/`。
- 验证：标签为 54 train / 18 holdout，ΔNLL 范围 `[-0.011271, 0.361022]`；三项局部特征已完成。decode sparse-NVFP4 需按 kernel 的 32-token shape 合约补零后裁回真实 80 token，重测已正常进入计算。
- 后续注意：补零只解决 wrapper shape 合约，误差统计始终只计真实 80 个 decode token；待第四项结束后再拟合并报告 holdout 指标。

## 2026-07-18 - 统一全部方法的阶段局部特征并完成质量拟合
- 开发目的：让 quality proxy 的 sparse、dense-NVFP4、Marlin-W4A16 特征均与当前 phase runtime 的实际压缩算子一致。
- 修改内容：扩展 collector 以调用真实 NVFP4 W4A4、canonical sparse BF16/NVFP4、Marlin W4A16 wrapper；完成两阶段共 8 张特征表。新增 phase-aware positive proxy（global + phase + method + layer bucket + fused type 的 ReLU 校准），使用 canonical real-vLLM phase-switch ΔNLL 作为标签。
- 影响文件：`scripts/collect_canonical_phase_local_errors.py`、`scripts/fit_phase_quality.py`、`llama2_7b_chat/local_errors/`、`llama2_7b_chat/reports/quality/`。
- 验证：所有特征表均为 16 行，局部误差量级符合 W4A16 < dense-NVFP4 < sparse-BF16 < sparse-NVFP4。旧 p54--p71 holdout 是窄域平衡模式，排序不具代表性；新增不读取 NLL 的 feature-space farthest-point 54/18 coverage split（`coverage_holdout.json`），其 holdout MAE/RMSE 为 0.010606/0.011953。
- 后续注意：coverage holdout 的低损失策略 ΔNLL 差异接近 NLL 噪声，Spearman 不应用作唯一准则；求解时要使用正向质量安全裕量，不能利用微小负 ΔNLL。

## 2026-07-18 - 复用原 prefill-decode roofline 速度口径并生成一轮候选 Pareto policy
- 开发目的：在 canonical sparse 仅改变权重质量、不改变 kernel 形状与运行时的前提下，不重建速度模型。
- 修改内容：新增 `scripts/solve_pareto.py`。它直接复用 debug 035 的 `KernelLatencyPredictor` roofline 原始延迟与 fresh-process E2E 单调校准，以 `sum prefill(M=32768) + 80 * sum decode(M=16)` 组合两个阶段；使用 055 canonical quality proxy 作为约束，并依据 action audit 禁止 decode `sparse_nvfp4`。
- 修正：将实体 dense-BF16 policy 固定为 `point_000` 及加速比锚点。之前代理模型中的零系数单元会让 zero-budget DP 错选快速非 BF16 动作，不能代替物理 baseline。
- 验证：生成 `pareto/predicted_points.csv` 与 10 个策略 JSON。`point_000` 为 128/128 个 prefill/decode dense BF16 模块，roofline 原始延迟 `2970.739 ms`；速度依据与 035 完全一致。
- 后续注意：055 早期 `--reuse-llm` 调速数据仅作诊断，不用于最终模型或汇总；下一步必须用 035 同款 fresh-process 基准 runner 对筛选点做 E2E closure。旧单调校准在最高速端没有键点，因此不应把其外推的平台值当作最终速度结论。

## 2026-07-18 - 候选点的真实 vLLM 闭环验证
- 开发目的：在不使用 `--reuse-llm` 的正式速度口径下，验证质量约束求解的代理预测与实际的单调趋势。
- 修改内容：新增 fresh-process speed closure 与 canonical phase-NLL closure runner，及 `summarize_pareto_validation.py`。
- 验证：4 个 NLL 点的实测 ΔNLL 为 `0.005826/0.012211/0.022388/0.212033`，与代理的从低到高损失单调趋势一致；最高速 `point_009` 实测 E2E 中位数 `2722.079 ms`、对 dense BF16 `1.825x`，5 次 CV `0.43%`。`point_004` 的中位数为 `3326.547 ms` (`1.493x`)，但 5 次中有一次 `4919 ms` 干扰样本，因此需以中位数并标注 CV `19.5%`使用。
- 后续注意：旧 035 单调 E2E 校准在高速端缺少锚点，对 `point_009` 的 `1.672x` 低估了实测 `1.825x`；最终图与论文数字应使用此 fresh-process 实测值，不应用校准器外推值。

## 2026-07-18 - 完成 canonical Pareto 下游任务验证
- 开发目的：对低/中/高质量损失的 5 个求解策略，完成真实 phase-hetero vLLM 生成下的 CNN/DM-1000、DialogSum-1500 和 IWSLT-333 评测。
- 修改内容：新增 checkpoint export、可恢复 shard dispatcher、metrics merger 及 task Pareto report builder；复用 035 的任务推理口径。任务期间遇到 vLLM 上一个 fresh process 释放 CUDA context 延迟，将任务评测的 `gpu_memory_utilization` 从 0.85 降到 0.75 后仅补跑缺失 shard。
- 验证：5 个 policy × 3 个 dataset 共 15/15 组输出的样本数已满足 `1000/1500/333`，空生成均为 0；已生成 ROUGE-L、BERTScore、SacreBLEU 汇总表与 3 张图。
- 产物：`llama2_7b_chat/task_quality/summary.csv`、`task_quality/report/summary.md`、`task_quality/report/pareto_{cnn_dm_1000,dsum,IWSLT}.png`。图中只将 point 004/009 标为实测速度；001/003/005 明示标记为 roofline 筛选速度，避免伪装为论文结论。
- 后续注意：point 004 的任务分数与实测速度已完整，但未有新的 055 NLL closure（表中为空）；这是报告完整性的可选补测，不影响下游任务结果。

## 2026-07-18 - 扩展 Pareto 闭环点并固定速度复测口径
- 开发目的：补足从低损失到高加速端的 Pareto 采样密度，同时避免 GPU 干扰和显存配置漂移污染速度结论。
- 修改内容：speed runner 支持显式记录 `GPU_MEMORY_UTILIZATION` 与 `RUN_GROUP`；以成对 dense-BF16 anchor 的固定 0.80 配置完成 point 001/002/003/005/006/008 的 fresh-process 速度采样。point 007 因外部显存占用导致 NVFP4 activation packing OOM，已判为无效且不纳入结果；0.75 的诊断试跑同样不纳入任何表图。
- 验证：0.80 口径下 point 001/002/003/005/006/008 的中位 E2E 延迟分别为 `4472.613/4492.765/4256.303/3941.508/3891.200/2791.210 ms`，paired dense-BF16 anchor 为 `4988.387 ms`；对应加速比分别约为 `1.116/1.110/1.172/1.266/1.282/1.787x`。point 008 的 5 次 CV 为 `0.234%`，point 003 有单次干扰样本，仍仅以中位数使用。
- 后续注意：0.80 extended curve 与此前 point 004/009 的 0.90 legacy speed closure 不能在未标注配置差异时混为同一正式曲线。已删除可复现的导出 checkpoint 释放空间；保留 policy、canonical weights、NLL labels、速度原始 JSON 和下游任务结果。

## 2026-07-18 - 完成扩展点真实 NLL 与下游任务闭环
- 开发目的：为新增的中间/高速 Pareto 点补齐真实 phase-switch NLL 和三项生成任务分数，提升曲线采样密度。
- 修改内容：在固定 100-block canonical vLLM NLL 口径下完成 point 002/006/008；导出三个可复现 checkpoint 后，以 batch 4、任务专用 `gpu_memory_utilization=0.75` 完成 CNN/DM-1000、DialogSum-1500、IWSLT-333 的 36 个分片。汇总器改为复用已有 `metrics.json`，并扩展 closure summary，显式区分 `util080_extended` 与 `util090_legacy` 的 paired dense anchor。
- 验证：新增点 ΔNLL 为 point 002 `0.009304`、point 006 `0.033365`、point 008 `0.131222`；全部 9 个 dataset-policy 输出达到预期样本数 `1000/1500/333`、空生成数为 0。统一任务表已含 8 个 policy × 3 个 dataset（24 行），并重绘三张图。
- 产物：`llama2_7b_chat/validation/closure_summary.csv`、`task_quality/summary.csv`、`task_quality/report/{summary.md,task_pareto_points.csv,pareto_*.png}`。
- 后续注意：任务 quality 曲线应按 `speed_config` 分面或明确标注；0.80 extension 与 0.90 legacy 不能无标签地连成一条正式速度曲线。point 006 的 IWSLT 单任务得分高于相邻低损失点，作为任务方差保留原值，不单独作为质量模型优越性的证据。

## 2026-07-18 - 统一 0.80 速度闭环并补齐 point 004 NLL
- 开发目的：消除旧 0.90 速度点与 0.80 扩展点混用的问题，完成所有入图 Pareto 点的真实 NLL 闭环。
- 修改内容：以固定 `gpu_memory_utilization=0.80` 重测 point 004/009；point 004 的 GPU 2 数据出现双峰，改用 GPU 7 独立五次稳定组。补测 point 004 的 canonical phase-switch 100-block NLL。更新 closure/report builder，使所有实测点标记为 `util080`，并以同一 0.80 dense anchor group 计算加速比；保留所有原始 JSON，仅从正式中位数/CV 中剔除超过该点中位数 15% 的明显慢异常，同时记录 raw/effective 样本数。
- 验证：point 004 的实测 ΔNLL 为 `0.015246`，其稳定 E2E speedup 为 `1.214x`；point 009 为 `1.856x`。最终 8 个入图点的有效重复数为 4--7，E2E CV `0.23%--2.47%`（point 003/009 的各一条异常慢样本已不参与统计）。所有 8 点均具真实 NLL 与三项下游任务分数。
- 产物：更新 `llama2_7b_chat/validation/closure_summary.csv`、`task_quality/summary.csv`、`task_quality/report/{summary.md,task_pareto_points.csv,pareto_*.png}`。
- 后续注意：不再要求同卡 anchor；相同型号 GPU、相同 runner/config 下允许跨空闲卡重复，只有明显异常慢值才排除。point 007 因速度 OOM 不入图，也无需补下游任务。

## 2026-07-19 - 为中段速度区间增加原模型的约束取样入口
- 开发目的：补齐 prefill-decode 实测曲线在约 1.3x--1.8x 加速比之间的稀疏区域，而不修改既有质量或速度模型。
- 修改内容：扩展 `solve_pareto.py`，支持在保持原 phase-aware quality proxy 与 035 roofline prefill+80×decode 公式不变的前提下，按目标 raw roofline speedup 求最小预测质量损失的 DP 解；生成独立的 `pareto/dense_speed/` 候选 namespace。
- 验证：目标 `2.20--2.48x` raw roofline speedup 产生 8 个策略组成不同的候选；其中 `dense_004` 与已有 OOM 的 point 007 等价，已标记为不测试。尚未启动 GPU 测试。
- 后续注意：目标速度只用于更密地读取同一预测前沿，不能替代实测 closure；后续仅对非重复、非 OOM 候选按速度→NLL→任务的顺序闭环。
