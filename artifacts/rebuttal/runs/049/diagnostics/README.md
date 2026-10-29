# Decode执行路径短诊断（不作为性能结果）

正式三轮完成后，对同一个DASO速度点、四种核心P+D负载使用Nsight。原数值/计时代码、权重及引擎配置保持一致；采集包含CUDA、OS runtime和Python GIL，所有文件保留。

**确认：4,740个Decode模型前向均对应CUDA graph replay。** 数量与15次生成×(63＋63＋63＋127)完全相等，未发现本次诊断中的Decode退回eager。该结论检查执行结构，不把带profiler的耗时当成正式速度。

正式速度仍有部分点跨轮变化超过5%。当前没有定位唯一原因；不能仅凭GPU未混跑或图已启用就称测速稳定，也不能把Graph之外的开销全部归因于CPU。主线程调度等待采样和Graph/外部流程统计保存在analysis.json，仅作为后续定位线索。

[分析数据与来源SHA](analysis.json) · [原始Nsight](daso_speed_trace.nsys-rep) · [SQLite](daso_speed_trace.sqlite) · [正式双场景结果](../../../responses/reviewer2/qwen_nearby_dual_scenario_run049/README.md)
