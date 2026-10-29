# Run015：统一执行路径后的重新分析

重测受运行时影响的速度、画像、分配和排名。相同有效map、canonical权重及数据协议的质量结果经过来源核对后复用；新map补齐对应质量任务。旧001–014归档完整保留，当前主表仍应依据其标注的实验编号解读。

已经完成：固定地址的阶段复用权重缓冲区、device-alpha FP4绑定数值检查、三份异构map多次生成/搬运一致性检查、80项新画像、62项validation、55个候选map速度校准及选择冻结。B1异构Graph/eager生成一致，B8/S4096 BF16显存检查通过。

运行顺序：改变map的完整任务与独立test → GPU0三轮统一主表测速 → 主表与固定旧map运行时对照 → 三个代表负载的两轮固定map迁移 → 开销/约束报告与归档。完整任务期间不并行本任务的正式速度campaign。

- [范围与验证计划](../../../../dev/plans/148_qwen3_graph_reanalysis_plan.md)
- [开发记录](../../../../dev/impls/148_qwen3_graph_reanalysis_impl.md)
- [冻结选择](configs/selected_reanalysis.json)：只用validation和校准速度，不使用独立test选择。
- [完整任务进度](results/full_quality_recovery.json) · [正式测速进度](results/formal_progress.json) · [负载迁移进度](results/workload_progress.json)。后两项在对应阶段开始时生成。
- [最终主表设计](configs/design.json)：17行。HAWQ主行固定使用.80×BF16校准E2E目标，并保留.95/.80/.65全部预算及原不可行的q配对点。目标通过线性代理转换，不等同于实际E2E保证。
- [数值检查](results/runtime_gate_device_alpha.json) · [FP4动态缩放检查](results/graph_bindings_numerical_gate.json) · [负载检查](results/workload_smoke_gate.json)。

新执行路径保留异步H2D与按层两阶段存储复用，不将完整两套权重常驻当作免费收益。主表E2E延续原协议不含请求前恢复，恢复与完整cycle另报。当前Graph阶段管理针对离线同阶段批次，不据此宣称任意在线mixed调度支持。

最终优劣依据统一运行时、实际质量、完整E2E和重复范围判断；不预设DASO一定胜出。初始化、Graph捕获、显存、准备与validation搜索成本分别报告。原始失败、不可行点、重复配置和不利结果均保留。
