# 新8bit接口实际运行检查（R5 gkP9 / R3 LrT2）

等待017核心完成，空闲卡运行已冻结共同FP4/INT8阶段map、SQ-INT8和FP8。B8/S128/O2、eager、1GiB KV，仅记录真实prefill/decode各Linear边界的dtype/shape/stride，验证BF16接口与内部压缩权重格式的区别。不作速度、质量或生产mixed调度结论。
