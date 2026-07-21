## 2026-07-16 - Implement real-vLLM prefill-only calibration bundle
- 开发目的：消除旧 Transformers/prepared-weight NLL 标签与真实 vLLM runtime 的口径差异，同时保持现有精度建模方法不变。
- 修改内容：新增 debug 046 的统一 72-policy 生成器、固定 WikiText prompt-logprob NLL runner、临时 phase checkpoint 导出/清理、可恢复多卡调度、标签合并、local+global 精度模型重拟合和诊断报告。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/`、`dev/plans/109_prefill_only_real_vllm_quality_recalibration_plan.md`。
- 后续注意：生成与静态检查已完成。真实 vLLM smoke 在当前执行通道中均于 engine 初始化后、首次 prompt-logprob 请求前被外部终止（仓库既有参考脚本复现同一现象）；因此未启动 72-policy 校准。`run_all.py` 应在具备稳定 vLLM 请求执行的 `vllm` 环境中先以 `--selection p00,p01` 验证，再继续全量。

## 2026-07-16 - Complete Llama2 real-vLLM NLL calibration
- 开发目的：以完整的真实 vLLM 标签验证重校准后的 prefill-only 精度模型。
- 修改内容：完成 Llama2 72 个 policy 的 100×2048-token NLL；在工作区磁盘不足时将自动清理的临时 checkpoint 放到 `/tmp`，并对资源冲突失败的点使用较低并发补跑。
- 结果：54 train / 18 holdout 的正系数 local+global 拟合，holdout MAE=0.121419 ΔNLL、RMSE=0.149305、Spearman=0.820433。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat/`。
- 后续注意：Llama3.1 的 `p00/p01` runtime smoke 已启动；Llama2 的实际 Pareto 求解仍应在确认该 holdout 质量满足需求后单独接入既有 solver。

## 2026-07-16 - Re-solve Llama2 prefill-only Pareto with real-vLLM NLL model
- 开发目的：将已验证的真实 vLLM NLL 精度模型接入原有 roofline/校准速度模型，完成新的约束优化闭环。
- 修改内容：新增 `solve_real_vllm_pareto.py`，冻结 `046` 质量系数并排除 intercept，生成 24 个非重复策略；新增临时导出、真实 NLL 与五次 prefill 速度验证脚本。
- 初始求解结果：低预算段优先选择 dense-NVFP4；sparse 仅在较宽质量预算下出现，符合真实 NLL 对 sparse 风险的建模结论。
- 后续注意：8 个覆盖全曲线的策略正在实际验证；完成后据实测 NLL/速度选择下游任务代表点并生成论文图表。

## 2026-07-16 - Add reproducible downstream evaluation and report generation
- 开发目的：在不保留大量策略 checkpoint 的前提下，完成新 Pareto 策略的真实 vLLM 下游任务与论文产物闭环。
- 修改内容：新增 `evaluate_pareto_tasks.py`，按策略临时导出、执行 real-vLLM lm-eval、持久化单任务结果并自动清理 checkpoint；新增 `build_paper_report.py`，汇总五次速度中位数、固定块 NLL、预测 NLL 与下游任务分数。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/`、`llama2_7b_chat/pareto/paper/`。
- 后续注意：点 018/020/022/023 已具备完整速度；点 000/008/012/015 正在补齐中断前缺失的测量次数，完成后即可确定代表点并启动全任务评估。

## 2026-07-17 - Complete measured Llama2 prefill-only Pareto bundle
- 开发目的：完成新 real-vLLM NLL 模型下，从约束求解到论文表格/曲线的实测闭环，并覆盖整个质量—速度范围而非仅高质量点。
- 修改内容：完成 8 个 Pareto 点的 fixed-100-block real-vLLM NLL、每点 5 次 prefill 速度、以及 WikiText、WinoGrande、ARC-Easy、ARC-Challenge、MMLU 五个任务；报告脚本现输出所有策略/统一方法表格和六张图（NLL 与五个任务）。
- 结果：每个点 `000/008/012/015/018/020/022/023` 均有 5 项下游任务结果。p012 为高质量候选（1.346×、ΔNLL=0.015）；高速 sparse 端点的实测 NLL/任务退化显著，作为完整前沿的高加速端点保留。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/llama2_7b_chat/pareto/{validation,paper}/`、`scripts/evaluate_pareto_tasks.py`、`scripts/build_paper_report.py`。
- 后续注意：新模型的稀疏主导区 NLL 低估较明显；若后续希望以这些点继续再求解，应补充或保守化该区域的校准，而不应将当前 p015+ 的预测值视为严格约束保证。

## 2026-07-17 - Add uniform real-NLL references to the Pareto plot
- 开发目的：使 NLL 图能够直接检验 mixed 策略与 uniform 压缩的真实支配关系。
- 修改内容：报告脚本接入 `046` 同一 fixed-block real-vLLM 校准中的 uniform ΔNLL（BF16、Marlin、dense-NVFP4、sparse-BF16、sparse-NVFP4），并同步写入所有方法论文表及 NLL 图。
- 结果：实测 dense-NVFP4 位于约 `(1.867×, ΔNLL=0.0538)`，确实支配当前 p015 及其后的 sparse 主导 mixed 点；该结论不再被图表遗漏掩盖。
