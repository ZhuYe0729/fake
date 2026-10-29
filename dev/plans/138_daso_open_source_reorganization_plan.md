# DASO Open-Source Reorganization Plan

## 目标

在仓库顶层建立可独立迁出的 `daso/` 项目，收敛论文最终实现、复现实验工作流、
结构化原始结果与额外视觉模型适配器。新项目不得依赖历史 `artifacts/debug`、旧
`fake/` 或 `scripts/` 路径；patched vLLM 与 CUTLASS wrapper 作为外部 backend。

## 步骤与验证

1. 建立规范项目骨架与配置接口。
   - 验证：editable install、模块导入和配置解析通过；没有机器绝对路径。
2. 抽取 canonical compression、质量/速度建模、DP solver、评测与报告核心。
   - 验证：使用合成 fixture 完成特征聚合、拟合、求解、policy schema 和指标测试。
3. 整理两模型、两场景、两平台及消融/appendix 的最终复现工作流。
   - 验证：所有脚本仅依赖新项目路径和显式外部 backend；preflight 能检查完整契约。
4. 归档论文结构化原始结果、policy、图表、表格和 provenance manifest。
   - 验证：hash、来源闭包与论文数值检查通过；可从归档数据重建论文数据产物。
5. 整理 MaxViT、DINOv3、MIRROR、FakeVLM 的独立适配代码和最小示例。
   - 验证：配置、模块选择与结果格式的无 GPU smoke 通过。
6. 在临时目录验证独立性并补齐文档。
   - 验证：复制 `daso/` 后不访问旧目录即可安装、运行测试并重建论文报告。

## 边界

- 不删除或重写现有实验目录。
- 不复制模型权重、canonical checkpoint、数据集、缓存、临时日志和失败结果。
- 不复制 patched vLLM 与 CUTLASS wrapper 源码，只保留接口、版本与安装检查。
- 根许可证暂记为 pending；保留完整第三方来源说明。
