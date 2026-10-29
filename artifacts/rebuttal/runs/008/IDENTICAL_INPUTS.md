# 人工 reviewer gkP9：identical module inputs的实际实现

以下按归档代码说明，避免将局部代理误差描述为完整压缩网络误差。

1. 使用原始BF16 Qwen3和固定训练校准token。先捕获第一个decoder layer的hidden输入及位置/attention kwargs，在CPU暂存；逐层运行未压缩的decoder，并将该层dense输出作为下一层输入。模型参数和传播输出都不替换成某个候选的输出。
2. 在Q/O/gate/down的forward hook中取得同一份输入Tensor；Q/K/V及gate/up按运行时的fused模块拼接参考权重。对该输入分别计算BF16参考`F.linear(x,W)`及各候选真实完整forward，候选内部的激活量化/平滑/输出缩放也被计入。
3. Hook不返回替换输出；所有候选读取相同dense-backbone输入。只累积每个候选的SSE、reference平方和及计数，最终误差为sum(SSE)/sum(reference_mass)，而非先算每块relative error再简单平均。中间激活按层在CPU暂存，并非必须把全模型激活永久存盘。
4. 新8bit激活scale使用128训练块的前2048 token；局部误差使用前16块。原五类方法误差来自001，新增三类8bit误差在002中追加。数据起点、scale和局部误差表已冻结；003/005不回调这些输入或拟合。

原代码：[prepare_8bit.py](../002/code/prepare_8bit.py)，尤其是capture、collect、error hooks及`inputs=outputs`；实际数值位于[local_errors.csv](../002/data/local_errors.csv)。

## 必须明确的decode近似

归档002把BF16 teacher-forced长序列的位置分成前2048个prefill token和后64个continuation token，计算后者的局部误差时，每条样本的64个continuation输入一起送入候选Linear。这并非在压缩后的KV路径中逐步重放；其候选调用M也不等于实际运行时B8逐步decode的M=8。

因此，identical inputs严格指局部候选对照使用相同的dense参考输入，并不意味着所有部署batch、激活量化分组和压缩后上游状态都相同。特别是NVFP4存在张量级activation scale，token分组差异可能影响其扰动。这是对代理模型适用范围的解释，尚不能单凭代码将其认定为所有质量违约的唯一原因。

最终全策略NLL标签和测试通过真实vLLM KV/teacher forcing获取，并有目标token及阶段调用核验。run005保持原质量代理，另外用目标B8的独立validation选择、再在test检查实际可行性；run006又直接验证实际mixed负载的B/S/O。不能把局部误差近似或预测预算写成全局质量保证。

## 英文答复的准确表述

We obtain reference activations from the uncompressed BF16 model and evaluate every candidate on the same module-input tensor; candidate outputs do not replace the activations propagated through the reference backbone. We aggregate squared output error relative to the reference output energy. The archived continuation features are collected from teacher-forced continuation positions and batched for local measurement, rather than obtained by replaying a compressed KV trajectory. This is an approximation used by the calibrated quality surrogate. We separately evaluate complete policies through actual runtime teacher forcing and report measured budget violations. The additional validation stage uses the target batch size and is distinguished from the original surrogate-only optimizer.
