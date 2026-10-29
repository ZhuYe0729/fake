## 2026-10-27 - 固定独立窗口审计设计
- 目的：复核当前探索点的质量预算，而非用旧test继续找点；三模型有限固定池、32/64 validation样本量对照、四真实预算及先选择后confirmation流程。
- CPU检查候选train窗口足够：Qwen880、Llama3.1 699、Llama2 466个对齐2112-token窗口不共享旧校准16gram，均足够固定抽取64+64窗口。
- Llama2旧P+D校准片段无法按整行在当前train流定位，多数内部片段/延续可定位；统一用保守16gram排除法，不臆测原语料错误或改动旧数据。
- 所有后续质量仍是公开WikiText train保留窗口上的复核，不虚称外部语料或新的下游盲测。

## 2026-10-27 - 三模型数据与固定池冻结并排队
- 在原校准样本之外，进一步排除完整旧validation/test分词流的共享16gram。剩余候选窗口Qwen819/Llama3.1 662/Llama2 297；各固定64validation＋64confirmation，seed168044/168045/168046，无源位置重叠。
- 固定池分别12/12/11个唯一map，去重后仍记录所有原标签；所有q/wP和LUT成本在自己的GPU输出前冻结。
- 044/045/046控制器会话90998/92744/20874，等待043完整审计；验证完成才写入选择与SHA，随后独立确认。32/64使用同64validation的前缀/全集，四预算全部保留。
- 每次validation后只删除自有可重建权重；confirmation重导出须完整safetensors SHA一致。重启不重写已有选择时间/父审计摘要，保留所有异常证据。

## 2026-10-27 - CPU流程预检通过
- verify_fresh_protocol.py重新核对三模型全部128个源窗口、输入SHA、16gram排除及validation/confirmation不交叉，结果protocol_check/VERIFIED.json。
- 用保留的独立小fixture验证8个32/64×四预算选择均符合预期，且整个选择过程在不存在任何confirmation文件时成功完成；验证BF16零保护值、guard筛选和延迟最小目标。
- fixture代码、JSON、CSV及日志均保留，不影响实际044/045/046数据或控制器。

## 2026-10-27 - 在GPU输出前进一步去除新fold间短片段重复
- CPU数据检查发现新validation/confirmation虽源位置不重叠，仍分别共享12/19/17个16gram hash；据数据本身、未观察任何新质量输出，固定新增cross-fold排除标准。
- validation和随机seed不变，按原打乱序列继续取与validation无共享16gram的64个confirmation窗口。两fold间共享16gram降为0。
- 原confirmation tensor、design/frozen_design、purge记录、CPU证明均存reference/data_before_crossfold_purge_*；旧CPU总证明另归档。新数据及SHA在自己的GPU开始前重新冻结；可由更新后的prepare_fresh_quality.py从同seed复现。
- 该调整只用于实验数据隔离，不使用模型结果挑样本，也不改旧原始实验。
