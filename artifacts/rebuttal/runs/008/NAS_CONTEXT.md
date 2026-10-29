# 人工 reviewer gkP9：硬件感知 NAS 的定位与答复材料

## 应承认的共同点

DASO与硬件感知NAS共享“离散动作、质量/性能代理、硬件成本下搜索”的思想。不能把这种通用形式或multiple-choice knapsack本身当作首次提出的贡献，也不宜用“设计层次不同”直接排除HAQ/HAWQ等混合精度分配方法。

| 原始工作 | 可核查的相关点 | 对DASO的启示 |
| --- | --- | --- |
| [ProxylessNAS, ICLR 2019](https://arxiv.org/abs/1812.00332) | 面向目标任务/硬件直接搜索架构，并使用实际硬件指标 | 目标硬件与实际时延早已是搜索目标；应引用并明确承继 |
| [FBNet, CVPR 2019](https://arxiv.org/abs/1812.03443) | 可微搜索设备相关的高效ConvNet架构，避免逐个完整训练所有候选 | 不应把用代理减少枚举成本写成DASO独有思想 |
| [BRP-NAS, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/768e78024aa8fdb9b8fe87be86f64745-Abstract.html) | GCN性能预测、模型二元关系与迭代数据选择；讨论层级代价相加的误差 | 必须实证模块预测与整模型误差，并说明是否计入真实运行时成本 |

这些是论文原始页面核查后的归纳，不是对所有NAS方法的完整分类。BRP-NAS年份/会议为NeurIPS 2020。

## 可支持的贡献表述

DASO固定预训练Transformer架构与权重来源，在同一真实后端中选择每个Linear模块及推理阶段的量化/稀疏表示。贡献应落在跨压缩方法的可部署动作空间、不同阶段/shape的实际完整forward成本、从局部扰动到策略质量的校准，以及执行不同格式所需的运行时集成。新增run001–003用Qwen3、真实W8A8/FP8、HAWQ-v3 PTQ移植和工作负载迁移补充证据。

HAWQ的PTQ移植不等于原CNN QAT/TVM流程；所有方法在同一候选kernel与设备/负载下的比较才有意义。NAS架构搜索的ImageNet精度或训练成本不能直接充当Qwen3推理对照。本文补充的混合精度分配对照更接近固定LLM部署问题，仍需在相关工作中明确NAS脉络。

run004补充模块预测留出误差、整模型残差、真实BF16模块接口、CPU/GPU/存储开销。run005单独验证精确与反向优化，不能把预测NLL或预测时延约束称为真实质量或SLA保证。run006的新mixed-token变体需要双表示驻留，其额外显存与原全局phase实现的限制应明确区分。

## 英文答复草稿（无需等待新增数值即可使用的概念部分）

We agree that deployment policy optimization is closely related to hardware-aware NAS, and will explicitly discuss ProxylessNAS, FBNet, and BRP-NAS. We do not claim novelty for the general use of quality/latency predictors or for the multiple-choice knapsack formulation. Our focus is selecting executable quantization and sparsity representations for a fixed pretrained Transformer, at module and inference-phase granularity, and accounting for their behavior in a shared inference backend. We have added a sensitivity-based mixed-precision allocation comparison and real INT8/FP8 candidates. The runtime, calibration cost, and measured feasibility of the resulting policies are evaluated separately from the discrete optimizer. We will also state the limitations of additive latency estimation and distinguish the original phase runtime from the additional mixed-token execution variant.

提交前应根据最终实验结果缩短、填入关键数字，并保留尚未解决的限制。不要写NAS是“不相关工作”，不要宣称已实现原论文没有的严格泛化保证。
