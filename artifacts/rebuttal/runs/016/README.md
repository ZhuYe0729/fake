# Run016：同候选分配器与阶段适配对照

[计划149](../../../../dev/plans/149_qwen3_matched_allocator_plan.md) · [实现记录](../../../../dev/impls/149_qwen3_matched_allocator_impl.md) · [当前进度](results/worker.json)

本轮先完成同候选、同线性延迟约束的DASO质量代理与HAWQ-style Hessian代理比较。prefill/decode同算子和阶段后端适配两组已同时预登记；不在观察结果后切换比较口径。原run015完整主表与增强HAWQ结果保留。

- 共同动作是BF16、plain INT8、NVFP4。plain INT8是DASO原七候选外的第八种动作，新增真实局部误差和48个训练期配置（每场景18 train/6 holdout），保持原质量模型结构与超参数。
- 主分配器对照使用同一个MILP和同一成本矩阵/上限，只更换质量目标；并列解对双方采用同样的最小预测延迟规则。三种延迟目标.95/.80/.65及中心Hessian符号敏感性控制全部预登记，真实不可行点不静默放宽。
- 另有同格式独立阶段选择与全八候选epsilon=.01/.04扩展，和主分配器比较分开报告，不把扩大搜索空间的收益全部归给代理模型。
- 新validation32、confirmation64窗口与历史005/015对应64窗口不重叠，保存starts/SHA。该benchmark以前有整体评估，不能称为完全未见过的新语料。先固定选择文件，再打开confirmation标签。
- GPU评测等待run015全部测量/审计结束；精度四卡动态领任务、每卡一个配置；正式速度单卡、NUMA固定、三轮3warmup+8timed与BF16首尾锚点。
- 完整任务使用本轮实际运行时重新评估，尤其避免将含动态4bit激活缩放的旧eager生成与可变EOS批次的Graph生成混为同一数值协议。新confirmation NLL和三轮速度先发布，完整MC/生成任务后续补齐。

[冻结设计](configs/frozen_design.json) · [训练任务](configs/training_jobs.json) · [主分配矩阵/配置池](configs/allocation_pool.json)（完成训练后生成） · [结果报告](REPORT.md)（完成对应阶段后生成）
