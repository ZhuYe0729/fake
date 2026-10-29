# Qwen3-8B 与 HAWQ-v3 分配基线补充实验

## 目标与已确认范围

- 在 Qwen/Qwen3-8B 上评测原有五种 uniform baseline、DASO，以及基于 HAWQ-v3 原始 Hessian 敏感度与 ILP 的精度分配基线。
- HAWQ-v3 复现重点为分配方案，使用本机真实支持的格式；优先验证 INT8/INT4，若 INT4 无可用执行路径则按用户授权替换为 FP4，明确标注 adapted allocator 和实际候选集合。
- 原始代码、来源版本、适配差异、所有成功/失败运行和质量/速度测量均保留；不以不理想为理由删除结果。
- 实验代码和输出放在 `artifacts/rebuttal/runs/001/`，按 reviewer 整理的最终材料保留实验来源。

## 步骤与验证

1. 冻结环境、原始 HAWQ-v3 代码、模型与数据协议，验证 kernel 支持。
   - 验证：记录 GPU/软件版本、代码哈希和实际矩阵形状；INT8/INT4/FP4 输出与量化参考一致，完成真实 forward benchmark。
2. 建立 Qwen3 独立实验入口，复用已有 compression/runtime；处理 36 层、Q/K norm、fused projections 和 non-thinking 模板。
   - 验证：BF16 与原模型输出一致；所有 baseline 的实际 backend、稀疏模式和 phase dispatch 可审计。
3. 复现 HAWQ-v3 的 Hessian trace 敏感度与 ILP；适配 DASO 并形成质量—延迟候选曲线。
   - 验证：Hessian 使用校准 NLL 的二阶导数而非 SparseGPT 输入协方差；小实例用穷举核对 ILP；分配成本、准备耗时与运行约束完整保存。
4. 统一评测原五种 uniform、HAWQ-v3 分配、DASO 和阶段统一 hybrid。
   - 验证：prefill-only 与 prefill-decode 使用同一 workload/样本/模板；保存实测 NLL、下游指标、绝对 TTFT/TPOT/E2E、峰值显存和重复测量。预算口径分别记录，最终按实测 frontier 比较。
5. 汇总可引用结果和局限，追加实现记录。
   - 验证：结果每行可追溯到策略、配置、代码和日志；失败或不支持的项目显式列出，不伪装为完成。

## 资源与清理约束

- 使用启动时空闲 GPU，正式速度测试保持设备独占；不结束他人的任务。
- 大型临时状态可放任务专属内存目录，运行目录记录重建方法。
- 用户允许清理确认不再使用且可重建的大型模型二进制；删除前记录精确路径、大小、用途与重建入口。
- 任何代码、日志、CSV、文档及其他结果证据均不清理；原始 Qwen3 权重与正式论文依赖不删除。
