# 154 — Qwen3固定prefill下的条件增量质量代理

## 目标与成功标准
- 用户将后续重点转为DASO本身的优化；按要求先完成plan153的373 GiB缓存清理，之后才开始本轮测试。
- 优先检验质量代理是否能正确排序decode替换：以真实SQ-INT8 prefill/KV及均匀SQ-INT8 decode为基准，校准分组替换引起的整模型ΔNLL，而不是继续沿旧局部MSE排序扩大候选池。
- 预测改进与部署改进分开验收：固定未拟合policy的MAE/近预算误判是否改善；最终选点在新validation/confirmation满足相同ε=.01/.04及保护规则，且同卡三轮速度是否优于强INT8基线。失败和退化均保留，不预设W4A16必须入选。

## 受控实现
1. 新增可选核心模块`daso/modeling/conditional_quality.py`，由BF16、SQ anchor、单组和双组的配对整模型NLL闭式标定增量及交互。保留signed增量，预测始终还原为相对BF16的ΔNLL。原QualityCalibrator源码和默认行为不改，以保留旧实验source-SHA校验。
2. 独立run029，只做Qwen3 B8/S2048/O64。prefill固定SQ-INT8，decode每组SQ-INT8/W4A16二选一；36层按原4个深度桶×4种融合Linear分成16组，每组9模块。
3. 冻结24个拟合扰动：16个单组、8个同深度桶内的QKV+O/GateUp+Down双组；另冻结8个结构化多组holdout，均不参与拟合。BF16/plain INT8/SQ-INT8/全SQ-prefill-W4-decode为固定控制，共36个校准map。
4. 校准使用原128训练/激活校准样本的前64块，全部运行真实Graph teacher-forcing，与后续相同B/S/O/runtime。policy holdout只是开发模型选择，不冒充新语料。最终validation128、confirmation128、两组速度输入从unused内部train窗口固定抽样，排除原128校准及028的208个新窗口；三者互不重叠。
5. 同一16组搜索空间比较旧代理anchor修正、原架构用相同新拟合标签重拟合、条件单组加和、条件单组+双组交互四模型。原架构固定4000steps/lr.03/reg.05；不扫超参。新模型在8个固定policy holdout上预先按MAE选定主版本，然后才开启最终validation。
6. 对每模型，在预测ΔNLL目标{0,.005,.01,.04}下枚举全部65536个组mask，取LUT成本最小者；mask去重，联合固定anchor形成最多20个候选。各模型的最终选择只使用本模型提案及共同anchor，不能把别的模型点的收益算给本模型。主版本由前一步holdout决定，不根据最终质量/速度反选模型。
7. 沿用真实validation mean+2SE+.005，BF16为0，预算.01/.04；同卡实测calibration E2E选点，cycle另报。先冻结所有版本选点后，主版本与固定INT8/SQ/BF16/full-W4控制才读取新confirmation；均值与保护值通过情况分列。无需扩展Llama或重跑六负载矩阵。
8. 主版本两预算及固定控制三轮正式速度，保持同输入/卡/NUMA/Graph/KV与BF16首尾锚点。新主版本map才补完整三生成，相同map且代码/数据协议一致的027/028 Graph结果带SHA复用。下游分数不反向参与选择。

## 数据、成本与资源
- 保存所有policy/data/code SHA、校准标签、未拟合policy诊断、枚举最优性检查、候选去重和追加校准成本。16组是一次有界诊断/改进分支，不宣称144模块全动作空间全局最优。
- 新增模块配真实有意义的CPU测试：BF16锚点还原、signed改善、交互项、未拟合组合预测、形状约束；旧modeling测试也须通过。
- 采用已有measurement共享/排他锁和GPU claim，最多3张空闲非GPU0卡做质量，GPU0排他测速；不停止任何外部进程，不改共享vLLM/kernel。
- 原始模型、canonical、校准与结果保护；只清理本轮可重建临时权重并记录，留足64 GiB。原质量模型/015–028封存结果均不覆盖。
