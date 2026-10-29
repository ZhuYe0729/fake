## 2026-08-05 - 独立 DASO 论文复现项目
- 开发目的：从历史实验仓库中收敛可开源、可迁移、可继续开发的论文最终代码与结果。
- 修改内容：新建 `daso/`，实现 canonical SparseGPT、质量/速度建模、单调 E2E 校准、DP 求解、真实 phase-vLLM 评测、下游任务评分、论文绘图和 provenance 校验；归档两模型、两场景、两平台、消融及 appendix audit 数据；增加 MaxViT、DINOv3、MIRROR、FakeVLM staged adapters。
- 影响文件：`daso/` 全目录，以及计划 `dev/plans/138_daso_open_source_reorganization_plan.md`。
- 验证：11 个 CPU 测试通过；归档 16 项论文数值检查通过；Llama2 prefill-decode 质量拟合、速度拟合和 2560-action DP 闭包重放通过；复制到 `/tmp/daso-standalone-validation-20260805-v2/project` 后可离线 editable install、测试并重建数据驱动论文产物。
- 后续注意：patched vLLM 和 CUTLASS wrapper 按用户要求仍是外部 backend；模型权重、canonical checkpoint 和数据集不入库；许可证待正式选择。
