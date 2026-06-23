## 2026-06-18 - Initial workflow scaffold
- 开发目的：为 FakeVLM 建立类似 Llama2 `018` 的 prefill-only 精度建模 + 速度建模 + Pareto policy + 实测验证流程。
- 修改内容：新增 `024_fakevlm_prefill_global_pareto` 实验目录、计划文件和脚本框架。
- 影响文件：`dev/plans/056_fakevlm_prefill_global_pareto_plan.md`，`dev/impls/056_fakevlm_prefill_global_pareto_impl.md`，`artifacts/debug/024_fakevlm_prefill_global_pareto/`。
- 后续注意：完整质量建模与验证需要在 GPU 节点运行；默认脚本提供 smoke/full 参数，不自动提交作业。

## 2026-06-19 - Local GPU smoke validation
- 开发目的：确认 `024` 流程能在本机 GPU 和 `cospaq` 环境下跑通。
- 修改内容：使用 GPU 7 跑通 local error 采集、stratified policy 生成、小样本质量验证、质量模型拟合、batch 16 cost/Pareto、dense/sparse policy prefill speed 验证、selected policy quality 验证和汇总。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/smoke/` 生成 smoke 结果。
- 后续注意：smoke 只覆盖 2 个模块和 2 个样本，结果仅用于验证流程，不代表最终 Pareto 结论。

## 2026-06-20 - Full FakeVLM Pareto run
- 开发目的：按 Llama2-7B `018` 类似流程完整跑 FakeVLM 的精度建模、速度建模、Pareto policy 选择和实测验证。
- 修改内容：完成 224 个可替换线性层的 local error 采集、61 个 stratified policy 的 FakeClue 全量质量验证、质量代理模型拟合、5 个 batch size 的速度成本表和 Pareto 优化、40 个 selected Pareto 点的 prefill 速度与 FakeClue 全量精度实测；将 `fit_quality_model.py` 内部拟合改为等价张量化实现以避免 CPU Python 循环过慢。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/sensitivity/`，`quality/`，`global_coefficients/`，`costs/`，`pareto/`，`validation/`，`summary/analysis.md`，`plots/speed_vs_accuracy.png`，`scripts/fit_quality_model.py`。
- 后续注意：最终汇总入口为 `artifacts/debug/024_fakevlm_prefill_global_pareto/validation/pareto_validation_joined.csv` 和 `summary/analysis.md`；质量模型 RMSE 为 0.0502768，selected quality/speed 验证均为 40/40 无缺项。

## 2026-06-20 - Report Pareto plots
- 开发目的：补齐可直接用于报告的 FakeVLM Pareto 最优图，并按 batch size 分别绘制。
- 修改内容：新增 `build_report_plots.py`，基于 full run 的 joined validation 数据生成 `report/final_fakevlm_report.csv`、`report/final_report_summary.md`，以及 batch 1/2/4/8/16 各自的 speedup-vs-FakeClue Pareto PNG/PDF；图中包含 predicted Pareto frontier、selected measured points、measured nondominated frontier 和 dense BF16 accuracy reference。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/build_report_plots.py`，`artifacts/debug/024_fakevlm_prefill_global_pareto/report/`。
- 后续注意：报告图横轴使用实测 E2E prefill speedup vs dense BF16，纵轴使用全量 FakeClue global accuracy；颜色表示替换的线性层数量。
## 2026-06-20 - Report plots with uniform baselines
- 开发目的：修正 FakeVLM report 图中缺少 uniform baseline 点、图形读法与 llama2-7b report 不一致的问题。
- 修改内容：更新 `build_report_plots.py`，将 `020_fakevlm_uniform_accuracy` 的 uniform FakeClue 精度与 `021_fakevlm_linear_hybrid_prefill_speed` 的 uniform prefill 速度合入 report CSV；按 batch size 重新绘制黑色 selected mixed Pareto policy 连线和红色 uniform baseline 方块。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/build_report_plots.py`、`artifacts/debug/024_fakevlm_prefill_global_pareto/report/`。
- 后续注意：uniform dense BF16 的速度使用本次 024 dense 实测点作为 speedup=1.0 的基准，其余 uniform 速度来自 021 的实测记录。

## 2026-06-20 - Switch FakeVLM quality modeling target to NLL
- 开发目的：修正 FakeVLM Pareto 质量建模误用 FakeClue accuracy drop 作为拟合目标的问题，使其与 Llama2 `018` 一样基于真实 NLL/loss delta 建模。
- 修改内容：新增 `validate_policy_loss.py`，对 FakeClue assistant answer tokens 计算 teacher-forcing NLL；修改 `fit_quality_model.py` 默认读取 `quality/stratified_loss.csv` 并拟合 `nll_delta`；更新 README workflow；新增 `launch_stratified_loss_jobs.sh` 用于多 GPU 分片跑 stratified policy loss。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/scripts/validate_policy_loss.py`、`fit_quality_model.py`、`launch_stratified_loss_jobs.sh`、`README.md`。
- 后续注意：正式 dense NLL baseline 已写入 `quality/stratified_loss.csv`；其余 60 个 stratified policy loss 任务在 tmux 会话 `fakevlm_loss_024` 中运行，每张 GPU 一个分片。完成后需要重跑 quality fitting、cost table、Pareto selection。

## 2026-06-20 - Rebuild Pareto from NLL model and launch validation
- 开发目的：完成 NLL/loss 建模修正后的 Pareto 重建，并启动新 selected policy 的真实速度与 FakeClue accuracy 验证。
- 修改内容：完成 61/61 个 stratified policy 的 teacher-forcing NLL 采集；重跑 `fit_quality_model.py`、`build_cost_table.py`、`optimize_pareto.py`、`select_validation_policies.py`；新增 `launch_selected_validation_jobs.sh`，先按 batch 1/2/4/8/16 分别占用 GPU 0-4 跑速度验证，速度结束后再按 8 个 GPU 分片跑 FakeClue accuracy。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/quality/stratified_loss.csv`、`global_coefficients/`、`costs/`、`pareto/`、`validation/selected_pareto_points.csv`、`scripts/launch_selected_validation_jobs.sh`。
- 后续注意：旧 accuracy 建模下的 validation CSV 已归档到 `validation/archive_20260620_165728/`；新验证在 tmux 会话 `fakevlm_validate_024` 中运行，当前 speed validation 已写满 40/40 行，accuracy validation 仍在运行。

## 2026-06-21 - Finish NLL-based validation report
- 开发目的：确认 NLL 建模后的 selected Pareto policy 真实验证完成，并重建报告产物。
- 修改内容：确认 `pareto_speed_validation.csv` 与 `validation_quality.csv` 均为 40/40 无缺项；重跑 `summarize_validation.py` 生成 `validation/pareto_validation_joined.csv` 与 `summary/analysis.md`；重跑 `build_report_plots.py` 生成 `report/final_fakevlm_report.csv` 和 batch 1/2/4/8/16 的 PNG/PDF 报告图。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/validation/`、`quality/validation_quality.csv`、`summary/analysis.md`、`report/`。
- 后续注意：本次 report 已对应 NLL/loss 建模后的新 Pareto policy；验证命令检查通过，包括 Python `py_compile` 和 shell launcher `bash -n`。
