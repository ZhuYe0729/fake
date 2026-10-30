# PRO 6000 · 完整 Qwen3-32B rebuttal 实验

**正式三方法完整结果与终审已闭合。** 参见[中文结果表及全部分项](../../responses/reviewer4/pro6000_qwen32b_run055/README.md)与[终审核验](results/final_audit_complete.json)。下方准备阶段文字保留为执行历史，实际完成范围以最终报告为准。

目标是补 R4 的完整30B级别模型证据；本轮独立于旧053调度与旧32B合成shape微基准。

- [计划184](../../../../dev/plans/184_pro6000_qwen32b_rebuttal_plan.md) · [开发记录](../../../../dev/impls/184_pro6000_qwen32b_rebuttal_impl.md)
- [模型逐分片身份与布局检查](results/model_identity.json)：17分片、707张量、64层、32,762,123,264参数，逐文件SHA；尚未与上游内容checksum比较。
- [完整BF16 HF可运行性验证](results/bf16_hf_feasibility.json)：全部64层、16步真实KV生成，仅作硬件/模型检查。
- [锁定vLLM BF16容量验证](results/bf16_vllm_feasibility.json)：B8/S2048/O64、BF16 KV、Decode Graph、固定8GiB KV容量；单GPU完整运行，峰值分配77,506,812,928字节。属于容量探针，不是正式对照速度。
- [数据采样身份](data/provenance.json)与[原Qwen协议逐token核对](data/comparison_with_run001.json)：重新用32B tokenizer编码WikiText，三个split与001逐token相同；128窗train校准、100窗validation、100窗test，样本起点与种子独立记录。
- [初始设计](configs/design.json)：256个fused Linear、B8/S2048、Prefill输出1与P+D输出64。当前处于准备/试运行，正式工作点和引擎身份尚未最终冻结。
- [phase fork二进制身份](configs/vllm_precompiled_binaries.json)、[FlashAttention随附源码](configs/vllm_precompiled_flash_attention.json)、[初始独立环境](configs/pip_freeze.txt)与[补齐评测包后的版本](configs/pip_freeze_with_tasks.txt)、[拓扑](configs/gpu_topology.txt)。实际fork为679527f，三处交接补丁与归档快照相同。

两套canonical状态、7项kernel数值验证、统一phase BF16 validation、新32B局部误差与PRO6000画像均已完成。HAWQ主迹P16/B4/S256已完成448个投影块，约72分钟；72套策略校准正在双卡推进。[状态快照](results/status.json)只依据已保存结果计数；正式质量/速度和下游任务尚未完成，不向中文总稿填入未测数值。

用户已将正式完整评测收窄为 **BF16、DASO、HAWQ-V3**；其他baseline不单独评测。uniform策略仅作DASO必要校准锚点，不另做其完整test/下游表。HAWQ小探针迹保留为试运行，主比较固定P16/B4/S256，每模块64个观察，使用signed目标。

新机器环境与旧归档的[逐项差异](configs/environment_comparison.json)单列。Python phase fork及迁移补丁已锁定，实际C++扩展来自官方0.11.1 wheel并记录SHA，与旧dev版二进制不能宣称相同；所有32B系数与本机画像均重新采集。ROUGE/SacreBLEU与lm-eval版本匹配旧协议，下游输入完整冻结。

入口为 `source artifacts/rebuttal/runs/055/code/env.sh`；GPU命令需在可访问设备的主机执行通道运行。各GPU使用独立锁；质量可分卡，正式测速在准备作业结束后串行进行。临时可重建策略权重将使用本轮专属内存盘目录；配置、代码、日志和测量结果始终保留。

[接续控制器状态](results/continuation_status.json)记录当前阶段；`code/continue_run.py`在校准与Prefill标签复用核验全部通过后接续三方法validation、冻结、正式test/速度与完整下游。异常会停止并保存日志。

用户确认尽量北京时间2026-10-30拿到结果，夜间有十几个小时。搜索已在DASO/HAWQ validation前扩展为[六档预算与四组阶段组合](configs/search_extension.json)，最终两档代表点经独立validation质量余量和单轮探索E2E选择，再冻结做正式三轮；所有探索点单独保留。质量闭环/test分双卡，探索/正式测速均单卡并持两卡锁。

验证反馈后，用同一冻结质量模型补充[0.003/0.005预测预算回退](configs/low_budget_refinement.json)，并[增加0.04中间报告预算](configs/reporting_budget_extension.json)，最终报告预算为0.02/0.04/0.07。全部原候选/不利点保留，正式测试仍在最终策略冻结后执行。
