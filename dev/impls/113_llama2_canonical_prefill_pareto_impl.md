# 113 Llama2 canonical prefill Pareto

## 2026-07-17 - Solver setup
- Added a 054 solver entry point that uses canonical sparse wrapper local errors and the fitted 054 softplus coefficients.
- Reuses the existing KernelLatencyPredictor roofline/calibration workflow and DP formulation; outputs are isolated under the 054 experiment.

## 2026-07-17 - Representative real-runtime validation launched
- 开发目的：验证 canonical 精度模型驱动的求解点，而非仅报告预测 Pareto 曲线。
- 修改内容：新增逐点导出、真实 vLLM NLL（100 WikiText blocks）和 5 次 E2E prefill 延迟测试脚本；已在 GPU 1--6 并发启动 p000/p005/p010/p015/p020/p024。
- 影响文件：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/validate_canonical_pareto_point.py`、`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/logs/pareto_validation/`。
- 后续注意：完成后汇总实测值、筛除异常延迟运行，并绘制含 uniform 基线的实测 Pareto 图。

## 2026-07-17 - Canonical validation and fair baseline closure
- 开发目的：覆盖 dense-NVFP4 邻域并消除 ours/uniform 的历史 runner 速度口径差。
- 修改内容：完成 p000/p005/p010/p015--p020/p024 的 100-block real-vLLM NLL 与五次 E2E 延迟；新增报告构建脚本。补充 p016--p019 后，p016 的实测 ΔNLL 为 0.0396（dense-NVFP4 为 0.0421）。同时启动所有压缩 uniform 方法的相同 phase-runtime 重测。
- 影响文件：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/build_pareto_validation_report.py`、`llama2_7b_chat/pareto/validation/`。
- 后续注意：uniform 同 runtime 延迟结束后重建最终图；再按曲线挑选少量点进行实际下游任务测试。

## 2026-07-17 - Real downstream validation launched
- 开发目的：验证 NLL Pareto 趋势是否延续到真实 prefill-only 下游数据集。
- 修改内容：将已有 task evaluator 扩展为可指定实验根目录与 canonical sparse 权重，canonical 路径下禁止追加 `--prune`；在 GPU 1--6 分别启动 p010/p015/p016/p017/p020/p024 的 WikiText、WinoGrande、ARC-Easy、ARC-Challenge、MMLU 完整评测。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_pareto_tasks.py`、`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/logs/tasks/`。
- 后续注意：待候选点结束后，选择仍有缺口的 uniform canonical 基线补测，并生成按数据集分面的最终 Pareto 图。

## 2026-07-17 - Mixed results completed; canonical uniform task closure launched
- 开发目的：完成可比较的真实任务曲线，而不混入旧 direct-prune uniform 精度结果。
- 修改内容：p010/p015/p016/p017/p020/p024 的 30 个完整下游任务结果均已完成；evaluator 支持显式 policy/label 后，启动 uniform dense-BF16、dense-NVFP4、sparse-BF16、sparse-NVFP4、Marlin-NVFP4 的 canonical 同口径评测。
- 影响文件：`artifacts/debug/046_prefill_only_real_vllm_quality_recalibration/scripts/evaluate_pareto_tasks.py`、`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat/pareto/validation/tasks/`。
- 后续注意：uniform 完成后生成五个任务的 speed-vs-metric 图，并从已验证曲线中标记论文候选点。

## 2026-07-17 - Full canonical task report completed
- 开发目的：交付不混入旧 runner、旧 direct-prune 精度的完整 prefill-only 论文产物。
- 修改内容：五个 uniform baseline 共 25 项任务完成（Marlin 的 MMLU 首次网络瞬断后单独成功重试）；报告脚本聚合 NLL、五次速度中位数与 55 项任务结果，输出统一 CSV、Markdown 表，以及 NLL 和五个任务的 Pareto 图。
- 影响文件：`artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/scripts/build_pareto_validation_report.py`、`llama2_7b_chat/pareto/paper/`。
- 后续注意：p017 是 dense-NVFP4 邻域的主要论文候选（更快且 ARC-Challenge 更高）；其它下游指标应在最终表中如实列出，不应仅以 NLL 宣称逐指标支配。
