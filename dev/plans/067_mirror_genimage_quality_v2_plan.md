# 067 MIRROR GenImage Quality V2 Plan

## 目标
- 用更多策略样本和 GenImage 部分样本 NLL 重新校准 MIRROR 质量模型。
- 保留当前 batch-shape-corrected 速度模型，不覆盖现有默认结果。
- 产物写入 `artifacts/debug/030_mirror_global_pareto` 下带 `v2_genimage` 后缀的目录/文件。

## 方案
- 生成约 200 个 V2 policies：dense baseline、bf16 baseline、单方法 ratios、module type/family targeted policies、controlled random policies。
- 非 baseline policies 使用 whole-model bf16 语义：未进一步压缩层标记为 `dense_bf16`。
- 对全部 GenImage split 使用 `sample-limit=192` 计算 CE/NLL，输出独立 CSV。
- 按 policy 聚合 GenImage weighted CE/NLL delta，重新拟合 quality coefficients。
- 用 V2 coefficients 重建 cost/Pareto，并选择 8-10 个点用于后续真实验证。

## 验证
- `py_compile` 覆盖新增/修改脚本。
- 检查 V2 policy 数量、quality rows 完整性、fit RMSE、Pareto unique points。
- 后续验证 selected speed/quality，并对比旧模型与 V2 模型。
