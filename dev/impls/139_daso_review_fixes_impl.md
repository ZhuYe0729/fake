# 139 DASO 正确性修复与开源收口开发记录

## 2026-08-05 - 修复求解器与完成开源审查收口

- 开发目的：处理 `code_review.md` 中确认的正确性、结果一致性和开源复现问题，同时保持 vLLM/CUTLASS 为外部后端。
- 修改内容：DP 预算采用保守离散并增加精确约束检查；质量模型 intercept 进入 action table、预算和预测结果；补充相应边界测试和 Llama2 prefill-decode 冻结数据回放。
- 修改内容：将旧论文表移入 `results/legacy_tables`，新增唯一 RTX PRO 6000 正式表；validator 改为来源 policy/metric 到论文单元格的结构化验证，并修正 overlap hidden ratio。
- 修改内容：对 1711 个文本结果文件执行机器路径审计和确定性脱敏，重建 1951 文件的 SHA-256 manifest；重绘流程同步更新论文源码可用的数据驱动图。
- 修改内容：固定 `lm-eval==0.4.12`，记录 vLLM/CUTLASS 仓库、revision、未发布 vLLM patch 和 release blockers；许可证继续等待所有者决定。
- 修改内容：为 MaxViT、DINOv3、MIRROR、FakeVLM 增加配置驱动的 dense loader/evaluator、数据适配和 selector smoke tests。
- 影响文件：`daso/src/daso/optimization/solver.py`、`daso/workflows/paper/`、`daso/results/`、`daso/extras/`、`daso/docs/`、`daso/tests/`。
- 验证结果：干净复制到 `/tmp/daso-standalone-validation-20260805-v4/project` 后，16 项 pytest 全部通过；18 项来源驱动论文数值检查、8 项结构检查、1711 个文本路径检查及全量 manifest 均通过；wheel 构建及四个 extras CLI 均通过。
- 后续注意：正式公开前仍需选择许可证，并把三处 vLLM 工作树修改发布为干净 commit/patch；缺失的手工论文图和部分 Llama2 逐样本生成证据没有被伪造。

## 2026-08-06 - 收紧论文表格的来源验证与确定性重建

- 开发目的：针对复审中确认的三个问题，补齐全部论文表格单元格的来源验证，隔离旧表，并兑现可重建表格的声明。
- 修改内容：新增 `main_table_sources.csv` 与确定性 `build_tables.py`，由结构化实验结果生成主表、RTX PRO 6000 表和最终消融表；重建脚本在图片和 manifest 之前先重建表格。
- 修改内容：validator 逐项核对主表 308 个、PRO 6000 表 84 个、消融表 16 个来源单元格，并校验 4 个 overlap 数值和 3 个生成表文件；篡改表格后即使重建 manifest 也会失败。
- 修改内容：将 pre-final `result_v3.tex` 移入 `results/legacy_tables`，补充 Llama3 decode uniform 的结构化速度来源；同时修正文档中的 extras 模块启动命令并增加 CLI 测试。
- 影响文件：`daso/workflows/paper/`、`daso/results/paper/tables/`、`daso/results/legacy_tables/`、`daso/tests/`、`daso/validation/standalone_validation.json`。
- 验证结果：在 `/tmp/daso-standalone-validation-20260806-v5/project` 独立副本中，21 项 pytest、3 张表重建、1952 个归档文件 manifest/路径检查、wheel 构建和 4 个 extras CLI 均通过。
- 后续注意：按本轮约定，没有扩展到 raw JSONL、extras 全压缩路径、后端内置或许可证决策。

## 2026-08-28 - 清理旧版 Llama2 导出模型

- 开发目的：释放旧版 `llama2_7b_018` 探索实验中不再被最终论文流程使用的模型磁盘占用。
- 修改内容：删除 4 个旧 uniform 与 6 个旧 heterogeneous max-speed `model.safetensors`，共 52.16 GiB；保留对应脚本、policy、manifest、历史测试结果、图表和文档。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/` 下 10 个被 `.gitignore` 忽略的模型二进制，以及本开发记录。
- 验证结果：当前论文与 DASO 归档对该旧目录无路径依赖；Llama2 最终来源仍为 054/056 canonical 轨道，论文归档验证和确定性表格检查均通过。
- 后续注意：plans 074--081 的历史实验若需原样重跑，应使用保留的导出脚本和配置重新生成 checkpoint；当前正式 `baselines/llama2-7b-chat`、`ours/llama2-7b-chat` 与 canonical states 未改动。

## 2026-08-31 - 清理已替代实验与 Qwen 可重建 checkpoint

- 开发目的：在不删除代码、实验结果、图表、日志、配置和文档的前提下，释放已被正式论文流程替代或未进入论文的可重建模型占用。
- 修改内容：仅删除 30 个明确核对过的 `model.pt` / `model.safetensors`：043 的 7 个旧 Llama3.1 checkpoint、003 的 4 个早期 Llama2 prepared state、037 的 4 个 refined-bridge checkpoint、034 的 3 个旧 uniform reference checkpoint，以及 Qwen3.5 0.8B/2B/4B 的 12 个 packed checkpoint，共 194,625,936,463 字节（181.26 GiB）。
- 影响文件：上述实验目录中的模型二进制，以及本开发记录；所有目录、README、脚本、policy、metadata、结果与报告均保留。
- 验证结果：五组待删模型计数均为 0；正式 baseline prepared state 8 个、canonical state 4 个、正式 fused export 12 个、056 论文 checkpoint 2 个均保持不变；Qwen metadata 12 份保留；DASO archive validator 与 3 张确定性论文表检查通过。
- 后续注意：043 的历史 checkpoint 若需重建，应直接按保留的 policy 调用 exporter；其旧 `materialize_all.py` 仅以 `phase_hetero_policy.json` 判断 checkpoint 是否存在，不能在只缺少模型二进制时直接使用。
