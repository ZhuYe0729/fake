# 066 MIRROR Layer-Heterogeneous Pareto Implementation

## 2026-06-29 - Initial workflow implementation
- 开发目的：为 MIRROR 增加层异构压缩的速度建模、精度建模、约束优化和 Pareto 报告流程。
- 修改内容：新增 `030_mirror_global_pareto` 实验目录、公共工具、policy runtime、local error、speed model、quality validation、quality fit、cost table、Pareto search、selected validation、summary/report 脚本和 smoke/full launcher。
- 影响文件：`dev/plans/066_mirror_global_pareto_plan.md`、`dev/impls/066_mirror_global_pareto_impl.md`、`artifacts/debug/030_mirror_global_pareto/`。
- 后续注意：模型和数据仍在传输，当前先完成代码与静态检查；完整 smoke/full 需等本机路径可用并有空闲 GPU。

## 2026-06-29 - Smoke preflight
- 开发目的：在本机后四张 GPU 视图下尝试启动 MIRROR Pareto smoke。
- 修改内容：修正 launcher 的本机 conda 初始化路径；确认静态编译通过；尝试 `run_smoke.sh`。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/run_smoke.sh`、`run_full.sh`。
- 后续注意：GPU 需要沙箱外运行；沙箱外已进入代码路径，但 `third_party/MIRROR` 为空，现有 `fake.models.mirror` loader 无法 import `models.mirror.build_mirror`。需要补齐 MIRROR 源码目录后再继续 smoke/full。

## 2026-06-29 - MIRROR source compatibility and smoke pass
- 开发目的：在 MIRROR source code 到位后继续验证 Pareto workflow。
- 修改内容：安装 `peft` 到 `cospaq`；修正 MIRROR source 对当前 Transformers DINOv3 结构的层路径兼容；让 MIRROR module selector 和 Pareto layer parser 支持 `backbone.dino.model.layer.*`；修正 smoke quality validation 未使用 `--sample-limit` 的问题。
- 影响文件：`third_party/MIRROR/models/mirror.py`、`fake/compression/modules.py`、`artifacts/debug/030_mirror_global_pareto/scripts/common_mirror_pareto.py`、`validate_policy_quality.py`、`quality/`、`sensitivity/`、`speed_model/`、`pareto/`、`validation/`、`report/`。
- 验证结果：`run_smoke.sh` 在 `CUDA_VISIBLE_DEVICES=4,5,6,7` 下跑通，实际使用可见 GPU 0（物理 GPU 4），生成 2 个 selected Pareto 点的 speed/quality joined validation 和报告图。
- 后续注意：smoke 只覆盖 2 个模块、少量样本和 batch size 2；正式结果需运行 `run_full.sh` 或分阶段用后四张卡并行跑 quality/validation。

## 2026-06-29 - Full speed model and parallel quality launch
- 开发目的：先完成 MIRROR 全层速度建模，并启动 stratified policy 精度建模阶段。
- 修改内容：完成 batch 16 全 224 个 MIRROR 可压缩 Linear 的 5 候选方法 local error 与 per-layer latency 表；重新生成 57 个 stratified policies；将 stratified quality validation 按 policy index 分成 6 份并绑定物理 GPU 0-5 并行运行。
- 影响文件：`sensitivity/module_method_local_errors.csv`、`speed_model/batch_16/module_method_latency.csv`、`stratified/quality_policies.csv`、`quality/stratified_quality.csv`、`logs/stratified_quality_gpu*.log`、`validate_policy_quality.py`。
- 后续注意：`validate_policy_quality.py` 的断点跳过粒度已改为 `key+benchmark+dataset`，便于补跑中途失败的 policy split；当前 quality jobs 仍在后台运行。

## 2026-06-29 - Quality modeling strategy note
- 开发目的：记录后续复用 MIRROR Pareto 流程时更高效的精度建模默认策略。
- 修改内容：在 MIRROR Pareto README 和 plan 中明确：质量模型拟合应优先增加 policy 组合多样性，并在固定分层 subset 上计算 CE/NLL delta；完整下游数据集指标主要保留给 selected Pareto policies 的最终验证和报告。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/README.md`、`dev/plans/066_mirror_global_pareto_plan.md`、`dev/impls/066_mirror_global_pareto_impl.md`。
- 后续注意：如果后续对其他模型复用该流程，推荐默认采用“更多策略 + 固定代表性 subset 拟合 + selected points 全量验证”，而不是“少量策略全部全量数据拟合”。

## 2026-06-29 - Finish selected Pareto validation
- 开发目的：完成 MIRROR batch 16 selected Pareto policies 的真实速度和真实下游指标验证。
- 修改内容：完成 57 个 stratified policies 的 513/513 行质量建模数据；拟合质量模型、构建 batch 16 cost table、优化 Pareto 并选择 8 个验证点；并行完成 8/8 selected policy 的真实 forward speed 和 72/72 split-level quality validation；生成 joined validation 和 report 图。
- 影响文件：`global_coefficients/`、`costs/batch_16/`、`pareto/batch_16/`、`validation/pareto_speed_validation.csv`、`quality/validation_quality.csv`、`validation/pareto_validation_joined.csv`、`report/`。
- 结果摘要：dense default point forward mean 99.080 ms；selected points 中实测最快为 P14/P17 附近约 51-52 ms，P21 的加权 balanced accuracy 最高约 0.8207；报告入口为 `report/final_mirror_report.csv` 和 `report/pareto_batch_16_speed_vs_bal_acc.png`。
- 后续注意：当前图展示 selected measured points 和 measured frontier；若用于正式报告，可进一步补充更多 P17-P24 附近点来细化高精度/高速度区域。

## 2026-06-30 - Pareto curve refinement and diagnosis
- 开发目的：分析当前 MIRROR Pareto 曲线观感异常的原因，并补测高压缩区域策略点。
- 修改内容：追加 P15、P16、P18、P19、P20、P22、P23 到 selected validation；并行完成新增 7 个点的真实 forward speed 和 63/63 split-level quality validation；重生成 joined validation/report；修改绘图为“所有验证策略散点 + 仅连接实测非支配前沿”，避免按预测 point order 连接 dominated policies。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/validation/selected_pareto_points.csv`、`validation/pareto_speed_validation.csv`、`quality/validation_quality.csv`、`validation/pareto_validation_joined.csv`、`scripts/build_report_plots.py`、`report/`、`summary/pareto_curve_diagnosis.md`。
- 结果摘要：batch16 现有 15/15 个验证点完整；实测最快仍为 P14/P19 附近约 1.93x，P20-P22 精度更高但速度回落；best balanced accuracy 更新为 P22，约 0.8263。
- 后续注意：线性速度模型在高稀疏/NVFP4 占比区间明显过于乐观，正式复用时应把 policy-level 端到端测速点用于校准速度模型，或增加 runtime overhead/后端切换等特征。

## 2026-06-30 - Effective BF16 runtime modeling fix
- 开发目的：修正 MIRROR 非 baseline policy 的速度/质量建模语义，使其与当前 runtime 的 whole-model bf16 行为一致。
- 修改内容：新增 effective method helpers；质量拟合时将 bf16 runtime 下的 `dense_default` 特征映射为 `dense_bf16`；Pareto 优化器将 P0 单独作为 fp32 dense baseline，P1+ 的候选集合去除 `dense_default` 并以 `dense_bf16` 为基础；归档修正前结果；复用已有 stratified quality 数据重新拟合、重建 cost、重新优化并完整验证 9 个 selected points。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/common_mirror_pareto.py`、`fit_quality_model.py`、`optimize_pareto.py`、`global_coefficients/`、`costs/`、`pareto/`、`validation/`、`quality/validation_quality.csv`、`report/`、`summary/pareto_curve_diagnosis.md`、`archive/before_effective_bf16_runtime/`。
- 结果摘要：修正后 selected batch16 为 9 个完整点；P1/P18/P19 的预测速度和实测速度已基本同量级，best speed 为 P19（约 1.912x），best balanced accuracy 为 P22（约 0.8255）；实测非支配前沿为 P22 与 P19。
- 后续注意：高 sparse/NVFP4 占比的 P20-P24 仍存在速度预测过乐观，说明还需要 policy-level overhead 校准；但 dtype 语义错位的问题已经修正。

## 2026-06-30 - Sparse NVFP4 ablation
- 开发目的：单独验证 `sparse_nvfp4` 是否是 MIRROR 高压缩区速度预测偏差的主要来源。
- 修改内容：为速度验证脚本增加独立 selected/output CSV 参数；构造 `dense_bf16 + topK sparse_nvfp4` ablation policies（K=0/4/8/16/32/48/64）；在 batch16 上完成真实 end-to-end forward speed 单测；生成 ablation 汇总。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/validate_pareto_speed.py`、`policies/sparse_nvfp4_ablation/`、`validation/sparse_nvfp4_ablation_selected.csv`、`validation/sparse_nvfp4_ablation_speed.csv`、`summary/sparse_nvfp4_ablation_speed.csv`、`summary/sparse_nvfp4_ablation.md`。
- 结果摘要：当前 per-layer 模型预测 top64 sparse_nvfp4 相对全 dense_bf16 应快约 5.18 ms，但实测反而慢约 4.16 ms；topK 增大后真实速度整体下降。
- 后续注意：`sparse_nvfp4` 是当前 MIRROR batch16 速度预测不准的主要来源之一；后续 Pareto search 应优先禁用 `sparse_nvfp4`，或加入基于 `count_sparse_nvfp4` 的强 policy-level overhead penalty。

## 2026-06-30 - Sparse NVFP4 single-layer marginal test
- 开发目的：回应 topK ablation 可能混入组合效应的问题，验证单个 sparse-NVFP4 Linear 的预测收益是否能迁移到整模型边际收益。
- 修改内容：构造全 `dense_bf16` baseline 和 12 个 one-at-a-time sparse-NVFP4 单层替换策略；使用 batch16、warmup 15、iters 100 完成真实 end-to-end forward speed 单测；生成单层边际对比。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/policies/sparse_nvfp4_single_layer_ablation/`、`validation/sparse_nvfp4_single_layer_selected.csv`、`validation/sparse_nvfp4_single_layer_speed.csv`、`summary/sparse_nvfp4_single_layer_ablation_speed.csv`、`summary/sparse_nvfp4_single_layer_ablation.md`。
- 结果摘要：预测收益最高的 12 个单层 sparse-NVFP4 替换，per-layer 预测每个应快约 0.085-0.110 ms；整模型 one-at-a-time 实测中位数反而慢约 0.145 ms，均值慢约 0.157 ms。
- 后续注意：偏差不只是 topK 组合造成；对 MIRROR batch16，`sparse_nvfp4` 的 isolated Linear microbench 不能可靠代表整模型边际收益。

## 2026-06-30 - Speed input shape fix
- 开发目的：修复 MIRROR per-layer latency 建模使用错误模块输入形状的问题，并重跑速度建模到最终 Pareto 报告。
- 修改内容：`collect_local_errors.py` 分离 local-error 输入和 speed 输入，local error 继续使用 `sample_limit`，latency benchmark 改为使用真实 batch 输入 `max_samples=batch_size`；该修复覆盖全部候选方法；归档修复前结果；重跑 224 层 × 5 methods 的 speed/local 表、quality fit、cost table、Pareto search、selected speed/quality validation 和 report；重跑 layer21 `up_proj` 的 sparse-NVFP4 细粒度 timing。
- 影响文件：`artifacts/debug/030_mirror_global_pareto/scripts/collect_local_errors.py`、`speed_model/batch_16/module_method_latency.csv`、`sensitivity/module_method_local_errors.csv`、`costs/`、`pareto/`、`validation/`、`quality/validation_quality.csv`、`report/`、`summary/pareto_curve_diagnosis.md`、`summary/sparse_nvfp4_module_timing_detail.*`、`archive/before_speed_shape_fix/`。
- 结果摘要：修复后 Pareto 不再选择 `sparse_nvfp4`；selected 8 个点完整验证，速度预测 RMSE 约 0.11；实测前沿为 P21（2.145x, bal_acc 0.7937）、P22（2.345x, bal_acc 0.7643）、P23（2.375x, bal_acc 0.7399）。
- 后续注意：绝对 predicted latency 是 Linear-only latency，不包含固定非 Linear 端到端开销；后续报告应使用 speedup ratio 和真实 selected validation。旧 sparse-NVFP4 ablation 结论应作为定位 shape bug 的过程证据，不再作为最终 method-level 结论。
