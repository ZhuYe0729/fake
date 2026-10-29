# 152 — Qwen3：保留基线并修正 decode 选点

## 问题与范围
- 用户要求解决 DASO 未选到有利 W4A16 decode 的问题。已确认旧精确解选中过全 W4A16，但真实 NLL 超过 ε=.01；手工 SQ-INT8/W4A16 的生成改善不能代替 NLL 可行性。
- 本轮为独立 run028，仅修正有实测依据的候选覆盖和局部精化，不改量化、kernel、原训练代理、ε 或保护公式；不覆盖 015/016/027 的已封存结果。
- 成功标准：保留可行 INT8 incumbent，产生并实测部分 W4A16 decode 候选，先冻结选择再测新确认数据，报告可行性及 E2E/cycle。若没有更好可行点，保留该结论。

## 冻结实验
1. 固定候选池：prefill ∈ {SQ-INT8, INT8, NVFP4} × decode ∈ {SQ-INT8, INT8, W4A16} 的九个全局组合；另加 BF16、全 NVFP4、全 W4A16。由全 SQ-INT8 和全 INT8 分别出发，只替换 decode，按冻结训练代理的增量误差 / 实测 LUT 节省排序，替换前 8/16/32/64/96/128 个模块。共24个唯一 map。
2. 数据：从原 Qwen WikiText train token 流抽取互不重叠的 128 validation、64 confirmation、8 calibration-speed、8 formal-speed 窗口；排除原128个训练/激活校准窗口的任何 token 区间。记录 start、seed、tensor/source SHA、内容去重断言。这是校准器未使用的内部留出窗口，不称为全新语料或官方新 test。
3. 原规则不变：paired mean + 2SE + .005 ≤ ε，BF16 的差值保护值为0，ε ∈ {.01,.04}。同时列32/64/128窗口保护值，但只用预定128窗口选择，不根据哪种样本量获胜改规则。
4. 对 ε=.04 下可行池和 INT8/BF16/旧NVFP4-SQ incumbent 进行新同卡 calibration；主选择最小实测 E2E，预声明另列包含恢复的 cycle 选择。每种目标在可行均匀基线存在时必须不慢于该基线的 calibration 值。
5. 冻结选择文件及 hash 后，才对选中策略、两种 INT8、旧 NVFP4/SQ、全 SQ/W4 对照测64新 confirmation。选中点与对照进行3轮 BF16 首尾锚点正式速度；E2E 与 cycle 同列。新增主选择的完整三生成任务用于结果描述，不能反向参与选择；已有完全相同 map 的027完整生成可带来源复用。

## 验收与资源
- CPU检查：池含所有声明的anchor、排名为真实正延迟收益、局部map只改decode、不同split/原校准样本无token区间重叠、可行incumbent不被丢失、源码及选择前后顺序可审计。
- 只新增 plan152/impl152 和 run028 文件。沿用审计过的 runtime 和 Graph bindings，不修改共享 vLLM 或旧实验。
- 精度每个短作业单独取得 measurement shared lock 和空闲非GPU0卡，最多3张；父线程020进入正式速度时主动让出。正式速度用 exclusive lock 和 GPU0，父线程020优先。
- 自有 XFS 导出权重每个阶段用后可清理并记录；只删可重建 model.safetensors，所有代码、配置、日志、结果、失败及文档保留。
