# 130 Llama3.1 prefill-only uniform-coverage refinement plan

## 目标

在不修改 058/061 主实验结果、运行时和建模方法的前提下，新建独立 debug 实验，补充 Llama3.1-8B-Instruct prefill-only 策略点。主要目标是在 ARC-Challenge 上找到能够实测包络 uniform Dense NVFP4 的 ours 点：

- measured E2E speedup >= 1.75736x；
- ARC-Challenge `acc_norm` >= 0.511092；
- 使用与 061 完全相同的 warmed phase-vLLM 速度口径、canonical sparse states 和真实 vLLM 下游评测。

所有尝试点均保留并汇报，不能只保留成功点。

## 实施步骤与验证

1. 冻结协议与建立独立实验目录
   - 新建 `artifacts/debug/068_llama31_prefill_uniform_coverage_refinement/`。
   - 引用 061 的 warmed speed protocol、058 的质量模型、057 的 canonical sparse states 和 060 的最终 uniform 结果。
   - 验证：生成 source manifest，记录源 policy SHA256、模型路径、runtime 与目标阈值。

2. 低成本复核既有候选点
   - 优先复测 058 的 `bridge_dense_nvfp4_104/120` 与 061 未完整闭环的 `point_008`。
   - 速度采用一个空闲 GPU 顺序测试，避免多卡同时测速引入 CPU/PCIe 干扰；每点一次 warmup、五次 timed requests。
   - 验证：每点均产生完整的五次原始延迟和 median，检查是否异常及是否达到 1.75736x。

3. 必要时加密原求解器质量预算
   - 若既有候选未覆盖 Dense NVFP4，在 `point_007` 与 `point_009` 之间增加细粒度质量预算，仍使用原始 local+global NLL 模型和 kernel latency+PAVA E2E 模型求解。
   - 先用真实速度与 real-vLLM NLL 筛选，不使用 ARC-Challenge 标签参与策略生成。
   - 验证：公开保存所有候选的预测/实测速度、预测/实测 ΔNLL 和方法计数。

4. 并行下游验证
   - 对速度满足或接近阈值、且 NLL 合理的候选并行运行 ARC-Challenge 全量评测。
   - 找到成功点后，再对最终非支配候选补齐 WikiText、WinoGrande、ARC-Easy 和 MMLU，以检查该点不是单一数据集偶然收益。
   - 验证：所有指标来自真实 phase-vLLM checkpoint；速度测试与质量测试严格分离。

5. 汇总与论文图
   - 输出包含全部 uniform、061 原有 ours 和 068 新测 ours 的机器可读总表。
   - 绘图时展示全部实测点，仅连接 ours 的真实非支配边界，并明确标记新点。
   - 验证：程序化检查每个 uniform 点是否被 ours 支配；报告 ARC-Challenge 之外各指标的覆盖情况及失败点。

## 边界与停止条件

- 不改变 runtime、激活量化、canonical sparse 权重、数据集或指标定义。
- 不根据 ARC-Challenge 逐层反向挑选 policy；下游任务仅用于最终验证。
- 若原模型约束下的加密候选仍无法覆盖 Dense NVFP4，则如实报告模型/离散 action frontier 的边界，不通过删除 uniform 点或修改数值制造包络。

## 完成状态（2026-07-23）

- 步骤 1--5 均已完成。
- 主成功条件已满足：`fine_q208` 在真实 E2E 速度和 ARC-Challenge 上同时严格优于 uniform Dense NVFP4。
- WikiText 与 MMLU 的跨指标检查未覆盖 Dense NVFP4，已作为限制完整记录，不影响本计划预先定义的 ARC-Challenge 绘图目标。
