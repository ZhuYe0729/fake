# DASO 代码整理复审报告

审计日期：2026-08-06

## 结论

本轮修复解决了求解器预算离散、质量模型截距、依赖固定、路径脱敏和部分论文图重建问题。核心 Python 包可以安装，CPU 测试、冻结数据求解回放和 wheel 构建均通过。

但当前 `daso/` 仍不能视为已经完整满足最初整理要求，也不适合宣称“完整论文复现已经开源”。主要未完成项是：主论文原始结果没有完整归档、论文数值校验覆盖严重不足、extras 只保留了 dense 演示和 selector 骨架、结果目录仍有旧表，以及精确 GPU 复现与许可证仍被阻塞。

## 仍需修复的问题

### 1. 高：主论文的原始结果文件没有完整归档

新的 `daso/results/paper/main_rtx5090/` 主要包含汇总 CSV、policy、部分 lm-eval JSON 和图片，但没有保留主表生成实验的逐样本生成输出和逐次速度结果：

- `.jsonl` 逐样本生成结果：0 个；
- `iterations.csv` 逐次速度结果：0 个。

旧实验目录中仍存在可用的原始结果。例如：

- Llama-2 prefill-decode 有 30 个逐样本 JSONL，约 27 MB；
- Llama-3.1 prefill-decode 的 `task_quality/results` 有 51 个逐样本 JSONL，约 70 MB；
- Llama-2/Llama-3.1 prefill-only 还保留了原始 lm-eval `result.json` 和多次速度测量文件。

`daso/results/paper/main_table_provenance_report.md` 自身也明确记录了这些文件的位置，并将逐样本 JSONL、metrics、运行配置和 iterations 视为表格的原始证据。

因此，当前归档满足的是“最终聚合结果和部分 provenance 已整理”，不是原要求中的“所有用于论文和 appendix 的原始数据结果文件都已整理”。应将仍然存在的最终表格点原始输出复制到明确的 `raw/` 子目录。确实已经历史丢失的三组 Llama-2 baseline 可以继续作为已知缺口单独说明。

### 2. 高：archive validator 不能证明论文表格正确

`daso/workflows/paper/validate_archive.py` 当前只从来源检查 8 个主表单元格，并把每个单元格分别与 canonical table 和 `wu.tex` 比较，合计报告为 16 项；另有 2 项 runtime-overlap 检查。

它没有覆盖：

- 主表其余绝大多数单元格；
- 整张 RTX PRO 6000 表；
- ablation 表中的大多数结果；
- canonical table 与论文表格的全量一致性；
- 图表使用的数据列和 policy 选择。

已在 `/tmp` 隔离副本中构造反例：将 `tables/main_table_final.tex` 中 Llama-2 BF16 ARC-E 从 `72.81` 改成 `99.99`，重建 manifest 后运行 validator，结果仍为 `ok: true`。

manifest 只能证明“生成 manifest 后文件没有变化”，不能证明数据正确。应建立完整的结构化映射：

```text
原始/结构化来源行 -> canonical table 单元格 -> wu.tex 单元格
```

主表、PRO 6000 表和 ablation 表的每个展示单元格都应覆盖。当前 PRO 6000 正式表本身没有发现错误：本轮独立复算了 42 个展示单元格，均与 `selected_results.csv` 一致；问题是现有自动校验无法保证这种一致性。

### 3. 高：extras 尚未保留之前实验的完整代码支持

当前 MaxViT、DINOv3、MIRROR、FakeVLM 的 evaluator 基本只加载 dense 模型并执行基础评测。压缩相关内容主要停留在 `select_compressible_modules()` 的模块选择规则和一个人工小模型 smoke 脚本。

旧项目中已有但没有整理进 `daso/extras/` 的关键内容包括：

- MaxViT CUTLASS NVFP4、Sparse BF16、Sparse NVFP4 loader/checkpoint/runtime；
- DINOv3 CUTLASS runtime、layerwise policy、hybrid 和 sparse loader；
- MIRROR CUTLASS hybrid/policy workflow；
- FakeVLM uniform compression、policy runtime、局部误差、Pareto 和速度/质量评测流程。

因此目前只能称为“额外模型 dense adapter 和 selector 骨架”，还不能复现之前对这些模型做过的压缩实验。考虑到原要求明确说后续再考虑如何融入主框架，合理做法是先把必要的历史实现以 staged workflow 形式完整放进 `extras/`，而不是现在就强行抽象统一接口。

现有 pytest 对 extras 也只覆盖配置解析和 FakeClue 文本评分，没有覆盖模型加载、数据发现、压缩替换或真实结果格式。`extras/smoke_selectors.py` 未纳入 pytest，且使用的是人工构造模型。

### 4. 中：extras README 给出的 CLI 命令不能运行

文档要求从 DASO 根目录执行：

```bash
python extras/maxvit/evaluate.py ...
```

MaxViT、DINOv3、MIRROR 和 FakeVLM 的直接脚本入口均会报：

```text
ModuleNotFoundError: No module named 'extras'
```

当前可工作的形式是：

```bash
python -m extras.maxvit.evaluate ...
```

应统一修改 README，或者让直接脚本入口自行加入项目根目录。`validation/standalone_validation.json` 只记录“四个 CLI help 通过”，没有记录实际命令，因此该记录不足以证明公开文档中的命令可用。

### 5. 中：`results/paper` 中仍有冲突的旧表格

`daso/results/paper/cross_platform_pro6000/result_v3.tex` 仍包含旧的 Llama-3.1 Balanced 数值：

```text
49.23 / 77.82 / 70.64 / 55.93 / 1.91
```

最终主表使用的是：

```text
51.45 / 79.08 / 73.80 / 66.50 / 1.59
```

与此同时，`results/paper/tables/README.md` 声称所有 superseded tables 都已移到 `results/legacy_tables/`。应将 `result_v3.tex` 移入 legacy，或者明确重命名为历史汇总并补 provenance 说明，避免被误认为最终论文结果。

### 6. 中：项目声称可以重建表格，但没有表格生成流程

根目录 README 声称项目可以重建 data-driven tables and figures，但 `workflows/paper/rebuild_paper_artifacts.sh` 实际只执行：

- 生成数据图；
- 将五张 PDF 复制到论文源码目录；
- 重建 manifest；
- 运行不完整的 archive validator。

脚本不会从结构化数据生成 `main_table_final.tex`、`cross_platform_pro6000_final.tex` 或 `ablation.tex`。应增加确定性的 table builder，或将 README 改为只声明“校验和打包现有表格”。

### 7. 发布阻塞：外部 GPU backend 仍无法精确复现

vLLM 和 CUTLASS wrapper 保持外部依赖符合原整理要求，但论文实验使用的 vLLM 最终状态尚未公开：文档记录了相对指定 revision 的三个未提交工作树修改。

当前 backend audit 只检查 exporter、predictor 文件存在，以及源码中是否出现 `phase_hetero_mytest`；preflight 虽然记录 git revision，但不会：

- 与文档规定的 commit 比较；
- 检查 dirty working tree；
- 校验三个关键文件的 hash；
- 验证未发布 patch 已应用。

因此一个错误或不完整的 backend 也可能通过 preflight。正式发布前应把三个修改发布为干净 commit/patch，并让 preflight 强制验证 revision 和关键文件 hash。

### 8. 发布阻塞：许可证和完整论文 PDF 仍未解决

`LICENSE_PENDING.md` 明确没有授予任何使用权，正式开源前必须选择许可证并检查第三方代码、结果文件及数据引用的再分发条件。

完整论文 PDF 仍缺少：

- publisher style；
- bibliography；
- `motivation.pdf`；
- `overview3.pdf`。

其中 `artifacts/paper/` 原始目录本身只有 `wu.tex` 和 `appendix.tex`，因此这些缺失并非本次整理新造成的问题，但发布说明不能将当前状态描述为可完整构建论文 PDF。

## 已确认修复正确的部分

- DP 求解器已经采用保守离散，并在返回前检查精确质量预算。
- 质量模型 `bias/intercept` 已进入 action table、求解预算和预测结果。
- Llama-2 prefill-decode 冻结数据可重新生成 2560 个 actions 和 8 个策略；未发现精确预算越界。
- 16 项 pytest 全部通过。
- wheel 可以构建，并可安装到项目目录外后正常导入 `daso`。
- `lm-eval==0.4.12` 已固定到 eval optional dependency。
- 当前归档 manifest、机器路径脱敏和文件完整性检查通过。
- 五张数据驱动论文/appendix 图可以在独立副本中重建并复制到正确的论文源码路径。
- 当前 RTX PRO 6000 正式表的 42 个展示单元格复算一致。
- vLLM/CUTLASS 源码没有误复制进 `daso/`，项目总体规模仍约 75 MB。

## 建议修复顺序

1. 补归档现存的主论文原始 JSONL、metrics、运行配置和逐次速度文件。
2. 将 validator 扩展为主表、PRO 6000 表和 ablation 表的全单元格来源校验。
3. 整理 extras 中真正用于旧实验的压缩 loader、policy 和评测 workflow，并增加测试。
4. 修复 extras CLI 文档，清理或迁移 `result_v3.tex`。
5. 增加确定性的论文表格生成脚本，或收缩 README 中的重建声明。
6. 发布 vLLM 最终 patch/commit，并强化 backend preflight。
7. 确定许可证；如需完整论文 PDF，再补齐论文构建资源。

完成前 3 项之前，不建议将当前目录描述为“论文代码和全部原始结果已经完整整理”；完成 backend 与许可证事项之前，不建议发布为正式的完整复现版本。

## 本轮验证记录

- `PYTHONPATH=src python -m pytest -q`：16 passed。
- 临时 wheel 构建、隔离安装和项目外导入：通过。
- Llama-2 prefill-decode action table/solver 回放：8 个策略，0 个预算违规。
- `rebuild_paper_artifacts.sh`：执行通过，但只重建图，不生成表。
- 当前 archive validator：通过；篡改未覆盖表格单元格并重建 manifest 后仍通过。
- extras 直接脚本命令：失败；`python -m extras.<model>.evaluate --help`：通过。

本报告只更新审计结论，没有修改 `daso/` 源码或结果文件。
