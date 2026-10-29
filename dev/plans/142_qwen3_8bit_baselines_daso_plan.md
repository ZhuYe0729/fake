# Qwen3-8B 8bit baselines 与 DASO 扩展候选

## 目标与约束
- 用户要求补齐 reviewer 提出的 FP8/INT8 baseline，并把 8bit 加入 DASO 后重新测试。
- 新建 `artifacts/rebuttal/runs/002`，保留 001 已冻结证据和全部代码、日志、CSV、文档。只清理本实验可重建模型二进制，记录清理与重建方式。
- 复用 001 的模型、校准/测试划分、任务输入、SparseGPT 权重、工作负载与固定 BF16 KV；搜索和拟合不使用测试结果。
- SmoothQuant 使用官方通道平滑公式，alpha=0.5；INT8 为 W8A8，FP8 为 E4M3 W8A8。Qwen3 适配、平滑范围、动态量化粒度和实际内核明确记录。
- 阶段与模块候选需要保持原始 BF16 函数等价，因此平滑放在候选内部（可融合进激活量化），计入全部运行时间。原生 FP8 与 SmoothQuant-FP8 分开标记。

## 执行与验证
1. 核对官方 SmoothQuant 和本地真实 INT8/FP8 内核，记录源版本；四类 Qwen fused Linear、prefill/decode 形状验证数值、有限输出和实测延迟。禁止假量化速度冒充真实低精度速度。
2. 用冻结训练校准数据收集激活最大值，准备 SmoothQuant-INT8、原生 FP8、平滑 FP8 baseline。完成 test NLL、同卡速度、显存/主机缓存和原有完整下游任务。
3. 在原五候选上加入 SmoothQuant-INT8 和 FP8；新增训练/独立 policy holdout 标签，复用可核验的旧标签；重新拟合原 QualityCalibrator，沿用预算与优化算法。新候选没有原 Roofline 模型时使用明确记录的实测 shape 成本，并保留原五候选同成本输入的控制组，区分候选扩展和延迟输入变化。
4. 冻结搜索策略后，评测 prefill / prefill-decode NLL 与同一物理 GPU 端到端速度；按预先确定预算位置选代表点完成五项 likelihood / 三项生成任务，并加入强 phase-uniform hybrid 控制。
5. 汇总实测质量—延迟、预算满足情况、候选使用比例、校准 holdout 误差、准备成本与适用限制；校验源与结果完整性，生成 reviewer 证据包，追加 impl 记录。

## 成功标准
- 两类 8bit baseline 确实运行实际低精度 GEMM；明确区分标准 SmoothQuant INT8 与 FP8 扩展。
- 扩展候选后的两场景 DASO 均有冻结策略、独立实测速度/NLL、完整代表任务和可重建元数据。
- 对比依据实际质量而非仅预测预算；如无提升或预算不满足，保留并报告。
