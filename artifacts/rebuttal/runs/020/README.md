# Run020：Llama-3.1的简单剪枝对照（R4 QboE）

按plan151继续。原Llama核心主表完成前控制器只等待；随后用空闲卡补Wanda和magnitude 2:4，并与原SparseGPT和BF16比较。

- 保留原7种unfused Linear选择、BF16稀疏格式与同一SparseBF16 kernel。
- Wanda按层传播前层已剪枝输出；从原数据加载代码恢复“过滤空行、关闭special tokens”，对齐原token总数、128个seed0采样起点和2048长度。原tensor未保留，不声称与不可用原tensor的字节哈希匹配。
- calibration数据、原SparseGPT状态与主表策略保持原样；本轮Wanda/magnitude权重单独保存。
- 64窗口Prefill/P+D NLL、完整MC/生成任务及三轮GPU0速度全部补齐后才标完成。
- 原SparseGPT/BF16引用封存run017，各速度使用各自campaign的BF16首尾锚点归一化，明确跨campaign来源。
- Llama-2的可选剪枝控制仍有一个token的原校准重建差异，未用近似样本冒充完全一致。

[冻结设计](configs/design.json) · [驱动](code/driver.py)
