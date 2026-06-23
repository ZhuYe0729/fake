# 060 FakeVLM Sparse BF16 Frontier Refinement Implementation

## 2026-06-22 - Prepare sparse BF16 neighborhood refinement
- 开发目的：填补 batch-16 P18 与 P22 之间过粗的实测折线，准确展示 uniform sparse BF16 附近的关系。
- 修改内容：选择器增加 `--include-points`；新增 P19-P21 独占速度、三卡全量 FakeClue 和带后缀报告的运行脚本。
- 影响文件：`select_validation_policies.py`、`run_sparse_bf16_frontier_refinement.sh`、batch-16 validation/report artifacts。
- 后续注意：旧 corrected-NLL 报告保留；新增报告使用 `refined_sparse_bf16` 后缀。

## 2026-06-22 - Launch P19-P21 full validation
- 开发目的：为 sparse BF16 uniform 附近补充三个真实 global-Pareto policy 点。
- 速度结果：P19/P20/P21 分别为 917.858/866.804/829.793 ms，对应约 1.431x/1.515x/1.583x；速度阶段在 GPU 0 串行独占运行，均为 224/224 replacement 且无 skipped module。
- 运行状态：全量 5000 样本 FakeClue 已在 GPU 0-2 各运行一个任务，日志无异常；预计约 85-90 分钟后完成并自动生成 `refined_sparse_bf16` 报告。
- 旧结果：精化前快照保存在 `archive_pre_sparse_bf16_refinement_20260622_022533/`。

## 2026-06-22 - Complete refinement and correct measured-frontier plot
- 完成情况：P19/P20/P21 全量 FakeClue accuracy 分别为 0.9864/0.9842/0.9790；11 点 joined report 和 `refined_sparse_bf16` 文件已生成，日志无异常。
- 结果解释：uniform sparse BF16 (1.521x, 0.9852) 实际支配 P20 (1.515x, 0.9842)，但 P19 (1.431x, 0.9864) 精度更高，因此 sparse BF16 是真实非支配 baseline，而不是仅由粗折线造成的视觉假象。
- 绘图修正：修正 measured frontier 的排序方向；弱化全部 selected-policy 连线，突出真实 mixed nondominated frontier，并标注 P19-P21；新增 `refined_sparse_bf16_v2` 后缀报告以保留 v1。
- 可视化收尾：调整 P20/P21 与 sparse BF16 的局部标注位置，最终报告图使用 `refined_sparse_bf16_v3` 后缀。

## 2026-06-22 - Add frontier-only v4 figure
- 开发目的：报告图中隐藏不属于真实 mixed Pareto frontier 的策略点。
- 修改内容：绘图脚本增加 `--frontier-only`；v4 只展示 measured mixed Pareto frontier 与 uniform baselines，CSV 仍保留全部实测数据。
- 展示调整：frontier-only 图不显示 P19/P20/P21 文字标号。

## 2026-06-22 - Prepare refined prediction-versus-actual update
- 开发目的：将 P19-P21 纳入 corrected batch-16 的质量、single-linear policy 汇总和 E2E 预测/实测对比。
- 修改内容：新增三卡增量 corrected-NLL launcher；三个 NLL 子任务全部成功后，生成独立的 `corrected_nll_batch16_refined_sparse_bf16/`，不覆盖原 8 点结果。
- 后续注意：single-linear 基础 profile 不重新测速；新增点只复用已有 shape/backend profile 进行 policy 级汇总。
- 启动状态：tmux 会话 `fakevlm_024_prediction_refine` 已在 GPU 0-2 分别启动 P19/P20/P21 corrected NLL；三项均完成 calibration 且日志无异常，完成后自动构建 11 点 comparison。

## 2026-06-22 - Complete refined prediction-versus-actual update
- 完成情况：P19/P20/P21 corrected NLL 分别为 0.678009/0.710203/0.788091；11/11 quality、11/11 E2E 和 12/12 single-linear comparison 全部生成，GPU 已释放。
- 输出目录：`prediction_vs_actual/corrected_nll_batch16_refined_sparse_bf16/`；原 `corrected_nll_batch16/` 仍保留 8 点版本。
- 指标摘要：NLL Pearson/Spearman 为 0.9968/0.9727；E2E MAPE 为 3.91%，Pearson 0.99974；single-linear 模型预测 MAPE 保持 6.72%。
- 结果解释：质量模型在 P19-P21 区域低估真实 NLL delta，但排序正确；E2E 对三个新增点高估约 2.97%-3.44%。
