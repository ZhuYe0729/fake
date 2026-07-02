# 068 MIRROR sparse_bf16 Additivity Debug Implementation

## 2026-06-30 - Offline additivity analysis and controlled sparse_bf16 debug
- 开发目的：定位 MIRROR `sparse_bf16` 精度加性模型对最终 CE/NLL 预测偏差的来源。
- 修改内容：新增 031 debug artifact；实现 sparse_bf16 聚合/残差分析脚本；生成 controlled sparse_bf16 policy 集合，覆盖 same-count random、low-error/speed order、layer bucket、family、module type；用 8 张 GPU 并行跑 40 个 controlled policy 的 GenImage partial NLL。
- 运行结果：controlled 验证完整覆盖 40 policies * 8 GenImage splits = 320 rows；同样 sparse_count=112 时 NLL range=0.08328（`lowerr_count_112` 到 `speed_count_112`），说明层选择/交互效应远大于当前加性 local cost 的解释能力；controlled 数据中 `predicted_quality_cost` 对真实 NLL delta 的 Spearman=0.576、RMSE=0.02083。
- 影响文件：`artifacts/debug/031_mirror_sparse_bf16_additivity_debug/scripts/analyze_sparse_bf16_additivity.py`、`generate_controlled_sparse_bf16_policies.py`、`controlled_sparse_bf16/quality_policies.csv`、`quality/controlled_sparse_bf16_genimage_quality.csv`、`tables/`、`plots/`、`summary.md`。
- 后续注意：建议下一步将 sparse_bf16 quality model 从纯 additive local error 改为 policy-level 模型，加入 sparse ratio/count、layer/type distribution、backend diversity/mixed-policy penalty，并用 controlled same-count 策略作为 held-out 验证。

## 2026-06-30 - Dense default AMP sanity benchmark
- 开发目的：补测 MIRROR 未压缩 dense-default 模型在 CUDA AMP autocast 下的速度和精度，确认与 FP32/BF16 baseline 的关系。
- 修改内容：新增 `amp_dense_default/scripts/validate_dense_amp_speed.py`；修正 MIRROR memory bank 中 AMP fp16 下 `masked_fill(-1e9)` 溢出问题，改用 `torch.finfo(attn_logits.dtype).min`；分 GPU 并行跑 speed、Chameleon quality、GenImage quality。
- 运行结果：AMP dense-default speed batch=16 mean=56.528820ms，vs FP32 dense speedup=1.7327，vs dense_bf16 为 0.9266x；Chameleon bal_acc=0.946393，GenImage weighted bal_acc=0.997920，质量与 FP32 dense 基本一致。
- 影响文件：`third_party/MIRROR/models/mirror.py`、`artifacts/debug/030_mirror_global_pareto/amp_dense_default/`。
- 后续注意：若要把 AMP dense 作为正式候选，需要在速度/质量报告中新增单独 baseline，不能覆盖原 FP32 `dense_default` baseline。

## 2026-06-30 - AMP baseline report refresh
- 开发目的：将 clean frontier 的相对加速图从 `uniform_dense_bf16` baseline 修正为 dense-default + AMP baseline。
- 修改内容：在 `report_keyfix_genimage_theoretical_clean_frontier_bf16_relative/` 内生成 `combined_report_amp_relative.csv` 和 AMP-relative PNG/PDF；同步覆盖默认 `combined_report_bf16_relative.csv` 与 `pareto_batch_16_speed_vs_bal_acc_bf16_relative.*`，并将旧 BF16-baseline 输出备份为 `*_uniform_dense_bf16_baseline.*`。
- 运行结果：AMP baseline forward_mean_ms=56.528820；`uniform_dense_bf16` 相对 AMP 为 1.0792x，`uniform_sparse_bf16` 相对 AMP 为 1.3990x，FP32 dense 相对 AMP 为 0.5771x。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/report_keyfix_genimage_theoretical_clean_frontier_bf16_relative/`。
- 后续注意：目录名仍含 `bf16_relative` 是历史遗留；以 `summary.md` 和 CSV 中 `speedup_vs_dense_default_use_amp` 列为准。
