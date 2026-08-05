# 127 Llama3.1 quality calibration-size sensitivity implementation

## 2026-07-22 - Fixed-holdout calibration learning curve
- 开发目的：验证完整 phase-aware 质量代理对 calibration policy 数量的敏感度，并同时量化真实 NLL 标签采集成本。
- 修改内容：新建065离线实验；固定最终 `coverage_holdout.json` 的18个holdout和5个anchor，按5/8/12/18/27/36/45/54规模重拟合；主线采用确定性feature-space farthest coverage，10组固定随机嵌套序列提供IQR；用057记录的逐policy elapsed time计算GPU-minutes；生成CSV、PNG/PDF、summary和论文文字。
- 验证结果：随机median holdout MAE从0.0614下降到0.0413，54样本端点在MAE/RMSE/Spearman上逐项精确复现064；预注册5%规则的经验拐点为45，54样本成本为48.6 GPU-minutes并处于平台区。
- 后续注意：`nll/prefill_decode.csv` 的历史 `split` 字段不是最终coverage划分，实验必须以 `policies/prefill_decode/coverage_holdout.json` 为准；不应为了让54成为数学拐点而重新选择holdout、随机种子或阈值。

## 2026-07-22 - 将学习曲线真实拓展到72个校准策略
- 开发目的：避免敏感度曲线恰好终止于主实验使用的54-policy点，验证更多校准策略是否仍能改善泛化误差。
- 修改内容：沿用039的确定性 `balanced_mixed` 生成规则新增18个calibration-only policy；审计确认与原72个策略无重复；用相同canonical sparse state、phase-vLLM后端和B8/S2048/O64/100-block协议采集真实NLL；在固定原18-policy holdout下增加60/66/72三个规模，并保持原54-policy模型不变。
- 稳定性处理：部分mixed policy在vLLM 0.80 KV-cache预留下触发权重materialization峰值OOM，重试仅降低 `gpu_memory_utilization` 到0.75/0.70并启用expandable segments；该参数只改变KV-cache预留，不改变NLL样本、policy或计算口径。
- 验证结果：54-policy端点与064的MAE/RMSE差异均小于1.5e-6，Spearman完全一致；median holdout MAE由54个策略的0.0413降至72个策略的0.0340，但成本由48.6升至71.4 GPU-minutes。按预注册5%规则，拓展后的经验拐点为72，而非54。
- 后续注意：应将54表述为主实验冻结的预算/精度工作点，不能宣称为渐近统计拐点；不能为了获得预期曲线而调整holdout、随机顺序或阈值。

## 2026-07-22 - 统一72-policy候选池抽样
- 开发目的：消除“原54个策略固定在前、新18个策略仅在54之后加入”造成的候选分布分段，使样本数量敏感度比较满足同一候选池抽样口径。
- 修改内容：将原54个训练策略与18个扩展策略合并为统一72-policy pool；10组随机嵌套序列和coverage诊断均从完整pool生成；冻结主实验54-policy模型单独拟合、验证并以星标展示。
- 验证结果：统一随机池在45/54/60/66/72个策略时的median holdout MAE为0.0403/0.0408/0.0392/0.0387/0.0340；冻结主模型保持0.0413并在1e-5容差内复现064。
- 后续注意：72完整池仍带来可观改善，不能通过隐藏72点或事后修改策略数据来制造54处饱和；论文应区分随机池统计曲线与实际冻结主模型。

## 2026-07-22 - 增加64-policy预算点与论文截断图
- 开发目的：让论文图聚焦实际考虑的校准预算范围，同时保留完整实验的可审计性。
- 修改内容：真实离线拟合64-policy随机子集；主图显示到64，另存完整72-policy PNG/PDF；CSV、Markdown表和验证文件继续包含全部点。
- 验证结果：64-policy随机median MAE为0.0394，IQR为[0.0373, 0.0423]，median标签成本为63.5 GPU-minutes；54到64处于相对平缓区间。
- 后续注意：主图的64预算上限必须在caption中明确；完整72点不可删除，且不能声称64之后已经统计饱和。

## 2026-07-22 - 清理论文图的截断与图例
- 开发目的：避免主图在64之后仍显示通往隐藏点的线段，并让图例术语更直观。
- 修改内容：论文主图的折线与IQR阴影仅使用不超过64的点；图例改为 `Median held-out MAE`、`Subset variation (IQR)` 和 `Calibration cost`；完整72-policy审计图不变。
- 影响文件：065 `scripts/build_report.py` 及重新生成的PNG/PDF。

## 2026-07-22 - 简化论文图标题与IQR表述
- 开发目的：减少论文子图中的重复标题，并让非统计读者直接理解误差阴影。
- 修改内容：移除64-policy论文主图标题；将 `Subset variation (IQR)` 改为 `Middle 50% of subsets`。完整72-policy审计图仍保留标题。

## 2026-07-22 - 恢复标准IQR图例
- 开发目的：同时保留直观含义和统计量的精确定义。
- 修改内容：图例恢复为 `Subset variation (IQR)`；IQR表示10组策略子集结果的第25到第75百分位范围，不能写成数学含义不同的variance。

## 2026-07-22 - 简化坐标轴与holdout术语
- 开发目的：减少图内术语负担，同时避免把policy数量误解为文本样本数量。
- 修改内容：图例改为 `Median MAE`，纵轴改为 `ΔNLL prediction MAE`；横轴改为 `Calibration set size (# policies)`。固定holdout定义继续在caption和报告中说明。

## 2026-07-22 - 精简横轴标题
- 修改内容：将横轴从 `Calibration set size (# policies)` 简化为 `Calibration set size`。

## 2026-07-22 - 明确主实验星标含义
- 修改内容：将 `Frozen main model: 54 policies` 改为 `Selected calibration setting (54)`，强调星号表示论文采用的校准配置而非另一种模型。

## 2026-07-22 - 去除星标图例中的重复数值
- 修改内容：将星标图例简化为 `Selected calibration setting`；具体数量由星号在横轴54的位置直接表达。

## 2026-07-28 - 调整论文主图纵横比

- 开发目的：改善 calibration-size sensitivity 图在单栏论文中的纵横比例，避免画面过于扁平。
- 修改内容：保持论文主图宽度与现有字号不变，仅将画布高度由 3.8 in 增至 4.2 in；完整 72-policy 审计图仍使用原尺寸。
- 影响文件：065 原始生成脚本与主图、067 consolidation 中的复现脚本及 `calibration_size_sensitivity.{png,pdf}`。
- 验证结果：目标 PNG 宽度保持 1288 px，高度由 804 增至 956 px；PDF 为单页 387.163 × 287.252 pt，字体已嵌入；图例、双纵轴标题和坐标刻度均无裁切。

## 2026-07-28 - 更正论文主图缩放方向

- 开发目的：按用户原意在固定宽度下适度压低图高，修正上一版反向增高的问题。
- 修改内容：将论文主图画布高度改为 3.5 in；宽度、字号、数据和完整审计图尺寸不变。
- 影响文件：与上一条记录相同。
- 验证结果：目标 PNG 为 1288 × 788 px，PDF 为单页 387.163 × 236.852 pt；图例、双纵轴标题和刻度均无裁切。

## 2026-07-28 - 简化校准成本单位

- 开发目的：缩短右侧纵轴标题并提升单栏图的直观性。
- 修改内容：将 `Calibration cost (GPU-min)` 简化为 `Calibration cost (min)`；单 GPU 采集口径保留在论文文字中说明。
- 验证结果：PNG/PDF 已重新生成，尺寸保持 1288 × 788 px 和 387.163 × 236.852 pt；PDF 文本检查确认旧单位已移除。
