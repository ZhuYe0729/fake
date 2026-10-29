# 143 — Qwen3 工作负载覆盖补充实验

## 目标与假设
- 回应 reviewer2 单一 B/S/O 与审稿元评审的 workload-adaptability 质疑，测试合理的交互/批处理、短/长提示和短/长生成负载。
- Qwen3-8B、RTX 5090，沿用 run002 真实 kernel、平滑设置和冻结质量模型；本轮不修改 DASO 质量模型或 2000-bin DP，避免把求解器改进和工作负载变化混淆。
- 新实验隔离在 `artifacts/rebuttal/runs/003`，001/002 的代码、日志、结果及文档不变。所有预先指定配置保留，包括失败、不利结果。

## 冻结设计
- P+D：(B,S,O) = (1,2048,64)、(4,2048,64)、(8,2048,64)、(16,2048,64)、(8,512,64)、(8,4096,64)、(8,2048,16)、(8,2048,256)、(1,512,256)。
- Prefill-only：上述 7 个不同的 (B,S)，O=1。逐项变化与一个实际交互组合，共 16 个场景。
- 统一 BF16 KV 6GiB，eager、无 prefix cache/chunked prefill，足够覆盖最大并发 token 数。与 001/002 的 3GiB 结果分开报告，原 B8/S2048/O64 点重测。
- Baseline：BF16、dense/sparse NVFP4、sparse BF16、NVFP4 W4A16、Marlin INT4、普通 INT8、SmoothQuant INT8、FP8；固定 phase-uniform 控制为 NVFP4/Marlin-NVFP4 和 SQ-INT8/Marlin-NVFP4。
- 冻结迁移对照：run002 七候选 prefill_02、decode_01，以及五候选 decode_04。分别在对应负载族测量，显式标为 fixed-policy transfer。
- 工作负载重分配：七候选 DASO 使用每个实际 shape 的完整 forward 成本，预算固定 0.01、0.07；质量拟合/局部误差复用002，所有策略先冻结再测测试集。
- HAWQ-v3 PTQ 移植：沿用001固定 Hessian 和量化误差、原 signed sensitivity + ILP，按新 shape 成本重新分配 INT4/INT8/BF16；延迟限制取对应 DASO 的预测 Linear 成本（低于可行最小值时显式记录并使用最小可行成本）。补充001 FP4控制策略迁移。普通 INT8 使用共同融合量化/缩放 kernel，需要先核验与旧实现的数值一致性；差异单独记录，不能把kernel改进归因于分配器。

## 执行与验证
1. 冻结配置、输入、依赖和继承哈希；真实 shape/kernel 检查与普通 INT8 数值对照 → 核验可运行和 lineage。
2. 同一 GPU3 测完整模块 forward 的多轮中位数，生成各负载策略 → 所有策略哈希冻结，记录候选比例/搜索成本/相同策略合并关系。
3. 同一 GPU3 顺序测正式速度，2 warmups + 5 timed；GPU0/4/7 分担独立质量测试，保持相同的原始测试 token → 验证固定输入/输出数量、phase transition、TTFT/TPOT/E2E/输出吞吐及内存。
4. 每配置 100 个冻结 WikiText 测试块；prefill 按原协议 B1 prompt logprob，decode 按目标 batch teacher forcing；报告实际 ΔNLL、预算违约与 paired block bootstrap 区间。质量模型跨 S/O 迁移是本轮限制，实际质量筛选仅用于分析。
5. 汇总全部负载、强基线、固定迁移/重分配对照及边界；作图、完整性审计，reviewer2/相关证据目录归档，追加 impl。

## 成功标准
- 16 个预定场景有完整 baseline 与指定 DASO/HAWQ 对照结果或明确失败证据；没有因结果不利而删除场景。
- 只在实际质量可比的测点间讨论速度优势；同时报告绝对延迟、输出吞吐、内存和预测预算满足情况，不把有限 sweep 表述成普遍最优。
- 原 001/002 manifest 可验证，代码/日志/CSV/文档全部保留。临时可复现权重仅在相关测量完成后释放并记录重建方法。
