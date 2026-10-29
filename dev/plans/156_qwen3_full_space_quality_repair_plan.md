# 156 — Qwen3完整候选空间的快速质量代理修正

## 目标与边界
- 用户要求针对原DASO没有选到合适配置的原因快速测试。恢复144个融合Linear模块的独立prefill/decode决策，每模块保留全部8×8组合；不固定prefill、不强制9模块同选。
- 先做中心负载B8/S2048/O64的可审计快速实验run031，比较冻结旧代理与轻量signed残差校正。主目标保持论文单批E2E；批间恢复另记，不作为主选择目标。
- 本轮先做NLL、候选预测和正式速度，不直接启动完整生成或扩展模型。旧主表/默认代理/封存结果保持原样。

## 实现与预定实验
1. 复用016全部144个全空间校准policy及029的26个拟合/8个policy holdout，统一取前64个原训练窗口。先验证token相同、BF16逐窗口兼容；每来源以其自身BF16配对。保留eager/Graph来源，不将协议差异隐去，不使用历史validation/confirmation标签拟合。
2. 用旧代理的可加局部贡献为prior，新增可选signed线性残差校准。特征是phase×method×4深度桶×4类型的方法占比和64个同模块prefill/decode组合占比；BF16特征为0，BF16质量预测固定为0。所有残差仍可分解到每模块的64个joint action，支持原一选一MILP。
3. 原policy train/holdout划分保留，029使用原fit/holdout。按有效map去重，任何holdout map均不得进入train；重合标签优先Graph来源。ridge仅在固定{.001,.01,.1,1,10}上按合并policy holdout MAE选择，在最终新validation之前冻结；不使用新test改模型。
4. 旧BF16锚定代理与校正代理分别在预测ΔNLL目标{-.02,0,.01,.04}下用同一精确一选一MILP求最小LUT延迟；允许负质量成本，保留求解gap/status和原始和，不使用旧DP的非正代价截断。每个模块的全部64动作入模；相同成本下若有并列，保持可复现tie规则。
5. 固定anchor为BF16、INT8、SQ-INT8、全SQ-prefill/W4-decode及030的90-W4配置。它们仅是共同备选/参照，不限制搜索空间。旧/新分别只用本模型提案和共同anchor选点，重复map去重，最多13个新validation map。
6. 从unused内部WikiText train窗口冻结validation128、confirmation64及两组速度输入8+8，排除原训练128、028/029/030全部窗口。若剩余不足，采用独立128validation+64confirmation优先，并明确速度输入可复用已有冻结速度输入；不得重用质量窗口或静默减少质量样本数。
7. 同一mean+2SE+.005规则，预算.01/.04。先新validation筛选，再同卡calibration E2E选择，冻结后对两模型选择及BF16/INT8/SQ anchor开独立confirmation。GPU0/NUMA0、同Graph/KV、3轮3warmup+5timed、首尾BF16锚点；报告实际通过/失败和测量波动，不按确认或正式速度重选。

## 验证与资源
- CPU测试验证signed残差、BF16锚点、全policy预测与joint-action求和一致、负代价MILP与小规模枚举一致，以及分组不限制模块选择。
- 实验前冻结数据/源码/标签/模型/提案SHA及集合关系；使用现有measurement/gpu/export锁、最多3张空闲卡，正式速度GPU0排他；不停止其他进程、不改全局GPU/CPU频率。
- 只清理本轮可重建临时权重，保留全部代码、日志、CSV、原始NLL、失败和文档。成功指更可靠地获得满足质量预算的快配置，不预设必须选W4或超过HAWQ。
