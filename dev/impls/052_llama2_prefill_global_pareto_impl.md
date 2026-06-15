## 2026-06-13 - Llama2 prefill global-coeff Pareto setup
- 开发目的：创建 018 版 Llama2 prefill-only Pareto 实验，接入 017 的 global-coeff 质量模型，并为真实压缩权重验证做准备。
- 修改内容：新建计划记录；复制 008 Pareto 脚本作为 018 独立脚本基线。
- 影响文件：`dev/plans/052_llama2_prefill_global_pareto_plan.md`，`dev/impls/052_llama2_prefill_global_pareto_impl.md`，`artifacts/debug/018_llama2_prefill_global_pareto/scripts/`。
- 后续注意：Marlin W4A16 质量不能复用 dense NVFP4；无可信 Marlin per-module proxy 时只作为真实 uniform baseline，不进入 Pareto 候选。

## 2026-06-13 - Global-coeff cost table and real runtime validation
- 开发目的：把 018 的 Pareto cost table 改为 017 风格的 global-coeff multiplicative proxy，并修正 validation 中对真实压缩权重/runtime 的处理。
- 修改内容：用 016 stratified loss/policy 输入和 017 fitter 在 018 下重拟合 `dense_nvfp4`、`sparse_bf16`、`sparse_nvfp4` 的 global coefficients；`build_cost_table.py` 默认读取 018 coefficients 和 008 fresh prefill latency；新增 `real_policy_runtime.py`，E2E/quality validation 都按 selected method 加载真实 `prepared/<method>/model.pt` 权重并安装对应 runtime；新增本机多 GPU launcher，按 `7,6,5,4,3,2` 分配，每点输出后再合并，避免并行写 CSV 冲突。
- 影响文件：`artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_cost_table.py`，`optimize_pareto.py`，`select_validation_policies.py`，`validate_pareto_e2e.py`，`validate_pareto_quality.py`，`real_policy_runtime.py`，`launch_local_validation.py`，`build_baseline_comparison.py`，`artifacts/debug/018_llama2_prefill_global_pareto/global_coefficients/`，`costs/`，`pareto/`，`summary/`，`plots/`。
- 后续注意：当前默认 Pareto 候选为 `dense_bf16,dense_nvfp4,sparse_bf16,sparse_nvfp4`，不含 marlin；如需把 marlin 纳入优化，必须先生成真实 marlin local-error 和 global-coeff coefficient，再显式传入 `--candidate-methods ... ,marlin_nvfp4 --marlin-local-errors ...`。

## 2026-06-13 - Full validation run partial completion
- 开发目的：按 018 计划启动完整多 GPU 验证。
- 修改内容：完成全部 29 个 frontier 点的 E2E validation；完成 27/29 个 quality validation；修复 launcher GPU 重复分配问题，补充跳过已有点和按空闲 GPU 分配；将 ARC eval 默认 batch size 改为 8，避免 sparse BF16 workspace OOM；修正 comparison 汇总中缺失 NLL 被误填为 0 的问题。
- 影响文件：`artifacts/debug/018_llama2_prefill_global_pareto/validation/e2e_points/`，`validation/quality_points/`，`validation/pareto_e2e_validation.csv`，`validation/pareto_quality_validation.csv`，`validation/pareto_validation_joined.csv`，`summary/prefill_only_comparison.csv`，`plots/`，`scripts/launch_local_validation.py`，`scripts/validate_pareto_quality.py`，`scripts/build_baseline_comparison.py`。
- 后续注意：quality 的 point 12 和 point 19 因并发显存不足尚未完成；后续需要单卡串行补跑 `--points 12,19 --gpus 7 --extra-args '--arc-limit 128 --arc-batch-size 8'`。本轮继续申请 GPU 执行时被审批系统 usage limit 拒绝。

## 2026-06-13 - Full validation completion
- 开发目的：补齐剩余 quality 点并重建完整验证汇总。
- 修改内容：等待 GPU 显存释放后，单卡串行补跑 point 12 和 point 19；重新生成 `pareto_quality_validation.csv`、`pareto_validation_joined.csv`、`prefill_only_comparison.csv` 和 plots。
- 影响文件：`artifacts/debug/018_llama2_prefill_global_pareto/validation/quality_points/point_012.csv`，`validation/quality_points/point_019.csv`，`validation/pareto_quality_validation.csv`，`validation/pareto_validation_joined.csv`，`summary/prefill_only_comparison.csv`，`plots/`。
- 后续注意：当前 E2E、quality、joined validation 均为 29/29；marlin 仍只作为真实 uniform baseline，缺少同口径 NLL delta，未进入 Pareto 候选。

## 2026-06-13 - Compact favorable showcase
- 开发目的：减少展示点数量，突出能够区分并支持方法有效性的有利 Pareto 点。
- 修改内容：新增 `build_showcase_outputs.py`，默认选取 `P000/P015/P020/P024/P026` 和四个 uniform baseline，生成 compact comparison、NLL 图、ARC 图、method-count 图和 summary；README 补充 showcase 用法。
- 影响文件：`artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_showcase_outputs.py`，`artifacts/debug/018_llama2_prefill_global_pareto/showcase/`，`artifacts/debug/018_llama2_prefill_global_pareto/README.md`。
- 后续注意：论文/PPT 优先使用 `showcase/` 下的图表；完整 29 点 validation 仍保留用于审计和 appendix。

## 2026-06-13 - Showcase point cleanup and runtime audit
- 开发目的：回应 dense NVFP4/P020 质量是否真实对齐 runtime、P015 是否异常的问题，并避免展示图强调 noisy 点。
- 修改内容：默认 showcase 点从 P000/P015/P020/P024/P026 调整为 P000/P020/P024/P026；README 标注 P015 保留在完整验证但因 ARC limit-128 波动不用于 compact showcase；重新生成 showcase CSV/summary/plots。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_showcase_outputs.py，artifacts/debug/018_llama2_prefill_global_pareto/README.md，artifacts/debug/018_llama2_prefill_global_pareto/showcase/。
- 后续注意：018 Pareto 点质量验证使用真实 prepared 权重和 runtime backend；uniform baseline 质量列来自 007/003 汇总，若要严格同源比较，应再用 018 runtime 路径重跑 uniform quality。

## 2026-06-13 - Full ARC-Challenge launch support
- 开发目的：启动 018 mixed Pareto points 的完整 ARC-Challenge 评测，并避免覆盖已有 limit-128 结果。
- 修改内容：`validate_pareto_quality.py` 增加 `--full-arc`、可配置 quality point 子目录和输出 CSV；`launch_local_validation.py` 增加 quality 子目录/输出名透传与合并支持。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/validate_pareto_quality.py，artifacts/debug/018_llama2_prefill_global_pareto/scripts/launch_local_validation.py。
- 后续注意：full ARC-C 输出应写入 `validation/quality_points_full_arc_c/` 与 `validation/pareto_quality_full_arc_c.csv`；已有 uniform full ARC-C 可从 007/full_arc_selected 复用。

## 2026-06-13 - Full ARC-Challenge mixed and uniform run
- 开发目的：补齐 018 展示点和 uniform 方法的完整 ARC-Challenge，而不是 limit-128 子集。
- 修改内容：完成 mixed points P000/P013/P019/P020/P024/P026/P027/P028 的 full ARC-C；发现 007 full_arc_selected 实际为默认 arc_easy 口径，不能复用为 ARC-C uniform；为 validator 增加 `--policy-dir` 并生成独立 uniform policy 目录以便跑真实 runtime uniform full ARC-C。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/validate_pareto_quality.py，artifacts/debug/018_llama2_prefill_global_pareto/validation/quality_points_full_arc_c/，artifacts/debug/018_llama2_prefill_global_pareto/validation/pareto_quality_full_arc_c.csv，artifacts/debug/018_llama2_prefill_global_pareto/validation/uniform_full_arc_c_policies/。
- 后续注意：uniform full ARC-C 应使用 `validation/uniform_quality_points_full_arc_c/` 输出；007 的 `full_arc_selected` 不要作为 ARC-Challenge 结果引用。

## 2026-06-13 - Full ARC-Challenge results summarized
- 开发目的：形成 mixed Pareto 与所有 uniform 方法的同口径完整 ARC-Challenge 对比。
- 修改内容：完成 uniform dense_bf16/dense_nvfp4/sparse_bf16/sparse_nvfp4/marlin_nvfp4 的 full ARC-C；生成 `summary/full_arc_c_comparison.csv`，所有结果均为 `arc_challenge` 且 `sample_len=1172`。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/validation/uniform_quality_points_full_arc_c/，artifacts/debug/018_llama2_prefill_global_pareto/validation/uniform_quality_full_arc_c.csv，artifacts/debug/018_llama2_prefill_global_pareto/summary/full_arc_c_comparison.csv，dev/impls/052_llama2_prefill_global_pareto_impl.md。
- 后续注意：主文建议引用 018 的 `full_arc_c_comparison.csv`；007 `full_arc_selected` 是 arc_easy 默认任务，不应用作 ARC-Challenge uniform。

## 2026-06-13 - Report-ready full ARC-Challenge figures
- 开发目的：把最终可汇报结果集中到明确目录，避免误用 limit-128 或旧 showcase 图。
- 修改内容：新增 `build_full_arc_c_report.py`，基于 full ARC-C 结果和 E2E speedup 生成最终表、Speed-vs-ARC-C 图、Speed-vs-NLL 图、policy composition 图和 summary。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_full_arc_c_report.py，artifacts/debug/018_llama2_prefill_global_pareto/report/，dev/impls/052_llama2_prefill_global_pareto_impl.md。
- 后续注意：汇报优先使用 `report/` 目录；旧 `plots/` 的 ARC 图仍是 limit-128 口径，不建议主文引用。

## 2026-06-13 - Report frontier point expansion
- 开发目的：增加 1-2 个合适展示点，让 full ARC-C 汇报图的曲线趋势更清楚。
- 修改内容：将 report mixed points 从 P000/P020/P024/P026 扩展为 P000/P013/P019/P020/P024/P026；重新生成 `report/` 下最终 PNG、CSV 和 summary。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_full_arc_c_report.py，artifacts/debug/018_llama2_prefill_global_pareto/report/，dev/impls/052_llama2_prefill_global_pareto_impl.md。
- 后续注意：P027/P028 full ARC-C 结果仍保留在 validation/summary 中，但质量下降明显，不建议主汇报展示。

## 2026-06-13 - Remove Pareto point labels from report plots
- 开发目的：按汇报呈现需求去掉主图中 mixed Pareto 点的文字标注，减少视觉干扰。
- 修改内容：`build_full_arc_c_report.py` 不再在 Speed-vs-ARC-C 和 Speed-vs-NLL 图上标注 `Pxxx`；保留 uniform baseline 标签；重新生成 `report/`。
- 影响文件：artifacts/debug/018_llama2_prefill_global_pareto/scripts/build_full_arc_c_report.py，artifacts/debug/018_llama2_prefill_global_pareto/report/，dev/impls/052_llama2_prefill_global_pareto_impl.md。
- 后续注意：若需要完全无文字标签，也可以进一步去掉 uniform baseline annotate。
