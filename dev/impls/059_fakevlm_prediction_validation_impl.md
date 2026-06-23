# 059 FakeVLM Prediction Validation Implementation

## 2026-06-21 - Prediction comparison workflow
- 开发目的：为 NLL 质量模型和 latency 模型补齐预测与真实结果对比。
- 修改内容：修正 validation dense loss 识别；新增 8 卡 selected-policy NLL launcher；新增质量、单 linear 和 E2E 三类对比汇总与绘图脚本。
- 影响文件：`validate_policy_loss.py`、`launch_validation_loss_jobs.sh`、`build_prediction_comparison.py`、`prediction_vs_actual/`。
- 后续注意：NLL 完整测试完成后 launcher 会自动生成带 `_prediction_vs_actual` 后缀的 CSV、Markdown、PNG 和 PDF；现有 `report/` 不会被覆盖。

## 2026-06-21 - Launch full selected-policy NLL validation
- 开发目的：采集 40 个 selected Pareto policy 的真实 NLL，并在完成后自动生成预测/实测对比报告。
- 修改内容：在 tmux 会话 `fakevlm_prediction_059` 中启动 `launch_validation_loss_jobs.sh`；先在 GPU 0 运行 dense baseline，随后自动切换到 8 卡并行分片。
- 后续注意：dense baseline 已稳定运行，GPU 0 利用率约 97%；质量 CSV 会在单个 policy 全量 5000 样本完成后追加，不会逐 batch 写入。

## 2026-06-21 - Complete prediction-versus-actual report
- 开发目的：确认完整测试结果并形成可审计的预测/实测对比结论。
- 修改内容：完成 NLL 40/40、单 linear 60/60、E2E 40/40 汇总；生成全部 CSV、JSON、Markdown、PNG 和 PDF；在 summary 中增加 raw NLL delta 分布和负相关警告。
- 结果摘要：单 linear 模型预测子集 MAPE 为 8.36%，E2E 全部点 MAPE 为 3.01%；质量预测与真实 NLL Pearson 为 -0.888，说明当前 NLL cost 在 selected policy 上未形成有效的校准预测。
- 影响文件：`artifacts/debug/024_fakevlm_prefill_global_pareto/prediction_vs_actual/`、`quality/validation_loss.csv`。

## 2026-06-21 - Diagnose and fix FakeVLM NLL label alignment
- 开发目的：定位质量预测与真实 NLL 负相关、且压缩后 NLL 反常下降的原因。
- 根因：LLaVA tokenizer 使用 left padding 并扩展 `<image>` token；旧代码将相对 token 长度直接当作绝对下标，实际 labels 覆盖 image tokens 而非 assistant answer。
- 修改内容：按 attention-mask active tokens 对齐 prompt/full 序列，去除 prompt 尾部 EOS 后做严格前缀校验，只标记 answer+EOS；新增 v2 loss definition，并让拟合与对比脚本拒绝 legacy loss。
- 验证结果：100 个 CPU 样本全部正确解码为 answer 且无 image token；32 样本 GPU smoke 中 dense NLL 为 0.500791，P30 为 1.112333，delta 恢复为 +0.611542。
- 后续注意：旧质量模型、质量 cost 和由其生成的 Pareto policy 均需重跑；既有速度模型和速度实测不受该 bug 影响。

## 2026-06-21 - Prepare corrected quality-model rerun
- 开发目的：先重跑 corrected stratified NLL 与质量拟合，验证模型方向后再决定是否进入完整 Pareto 验证。
- 修改内容：新增 `launch_corrected_stratified_loss_jobs.sh`；自动归档 legacy NLL 与 coefficients，运行 dense baseline、8 卡 stratified shards 和质量模型拟合。
- 后续注意：本阶段不会重建 Pareto 或启动 FakeClue accuracy；预计墙钟约 1–1.25 小时。

## 2026-06-21 - Launch corrected quality-model stage for batch-16 follow-up
- 开发目的：为后续仅验证 batch 16、并供 `025_fakevlm_pareto_search_audit` 使用的新 reference frontier 重建质量模型。
- 修改内容：旧 invalid NLL 与 coefficients 已归档到 `archive_invalid_nll_20260621_142347/`；tmux 会话 `fakevlm_corrected_quality_059` 已启动 corrected dense baseline，后续自动运行 8 卡 stratified shards 和拟合。
- 后续注意：完成并确认拟合方向后，只重建和验证 batch 16 的 8 个代表点，不运行其他 batch 的 FakeClue accuracy。

## 2026-06-21 - Rebuild corrected batch-16 frontier
- 开发目的：生成供 `025_fakevlm_pareto_search_audit` 使用的 corrected batch-16 reference frontier。
- 修改内容：corrected quality model Pearson/Spearman 为 0.760/0.805；归档旧 batch-16 cost、Pareto 和 selected 文件；仅重建 batch 16，得到 31 points、25 unique policies，并选出 8 个代表点。
- 新增：`launch_corrected_batch16_validation.sh`，串行执行独占速度测试、corrected NLL、多卡 FakeClue accuracy 和带 `corrected_nll_batch16` 后缀的报告生成。
- 后续注意：全 sparse NVFP4 的真实 NLL delta 2.418、模型预测 1.005，存在高压缩区欠拟合，最终报告需保留该风险。

## 2026-06-21 - Complete corrected batch-16 validation
- 开发目的：完成 corrected batch-16 reference 的真实 NLL、速度和 FakeClue 验证，并交接给 `025`。
- 修改内容：8/8 speed、NLL、accuracy 均完成；生成 `corrected_nll_batch16` report 和独立 prediction-vs-actual 子目录；更新 `025` reference_024 注册。
- 结果摘要：NLL prediction Pearson/Spearman 为 0.998/0.929；single-linear model-prediction MAPE 6.72%；E2E MAPE 4.17%；P22/P25 speedup 1.604x/1.635x，accuracy 0.9484/0.9528。
- 兼容修正：`025` neighborhood parent 选择在旧硬编码 point 不足 4 个时，改为从当前 selected frontier 均匀选择 4 个父点。

## 2026-06-21 - Prepare corrected 025 non-random validation
- 开发目的：验证 corrected quality model 生成的 `025` 搜索策略，同时跳过 random 搜索并保留旧结果。
- 修改内容：为多卡验证启动器增加 `--families` 筛选，并修正 `skipped_existing` 状态计数；本轮限定 `neighborhood,suspicious,reference_024`。
- 后续注意：旧 validation、logs 和 summary 在启动前移入带时间戳的归档目录；速度测试保持每张 GPU 同时仅一个任务。
- 运行入口：`scripts/run_corrected_nonrandom_validation.sh` 固定运行 38 个非 random 策略，并在全部成功后自动汇总。
- 启动状态：旧结果共 227 个文件已归档到 `archive_pre_corrected_nonrandom_validation_20260621_220822/`；tmux 会话 `fakevlm_025_corrected_nonrandom` 已在 8 张 RTX 5090 上启动首批 8 个 neighborhood 策略，未见报错或 OOM。

## 2026-06-22 - Complete corrected 025 non-random validation
- 开发目的：确认 corrected 025 搜索审计完整结束并检查结果可靠性。
- 完成情况：38/38 成功、0 失败；包含 20 个 neighborhood、10 个 suspicious、8 个 corrected `reference_024`，没有混入 random；自动生成 8 个 searched frontier 点及完整 summary。
- 一致性检查：8 个 reference 的 E2E 重测相对 024 corrected 实测偏差均在 ±1.30% 内；日志无 traceback、OOM 或 NaN，任务结束后 8 张 GPU 均空闲。
- 结果摘要：在相同 1000 样本子集上，4/8 个 reference 被搜索策略支配，最大 E2E latency 改善 25.72%；`suspicious_003` 为主要高精度候选。
- 后续注意：025 accuracy 使用 20% 固定子集，候选策略在形成最终结论前仍需全量 5000 样本精度复测。
