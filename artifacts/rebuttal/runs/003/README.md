# Qwen3-8B 工作负载覆盖补充实验（003）

目的：回应 reviewer2 对单一 batch / 输入长度 / 输出长度的质疑，并检查按工作负载重新分配是否有收益。

已在测量前冻结9个 P+D、7个 prefill-only 场景；B=1/4/8/16、S=512/2048/4096、O=16/64/256（非笛卡尔积，中心逐轴变化加一个交互负载）。
共267个预定比较项，按完整 method-map 合并为47个模型策略；包括9种 uniform、两个 phase-uniform、原策略迁移、DASO 按 shape 重分配、HAWQ-v3 PTQ 移植按 shape 重分配及FP4迁移控制。

- 配置：[design.json](configs/design.json)；策略：[evaluation_points.json](configs/evaluation_points.json)。
- 进度：[progress.json](results/progress.json)；运行中数表：[summary_current.csv](results/summary_current.csv)。
- 最终测速：物理 GPU0，同步同质 batch，统一6GiB BF16 KV，2 warmups + 5 timed iterations，eager，无 prefix cache/chunked prefill。GPU3被外部任务占用后全部统一重测；原GPU3部分结果保留为diagnostics，见[协议修订](configs/measurement_card_change.json)。
- 质量：100个冻结 WikiText 测试块，prefill按原协议 B1 prompt logprob，decode按目标B teacher forcing；原002质量模型不修改，明确检验跨S/O迁移。
- 普通INT8的激活量化与输出缩放使用融合实现，权重量化不变，和HAWQ共用；初始24项数值检查通过，最大relative MSE约2.04e-6，全部NLL重新测量。
- 原001/002保持不变；本轮不声称解决 mixed continuous batching 或证明严格质量保证。暂未修改原DP网格，避免混淆变量。

重现：环境沿用002。请在独立副本/新编号目录复测，保留本归档。保持策略冻结时直接使用已归档配置、模块profile和数据；从零生成策略则先执行 `code/setup_run.py`、GPU3上的 `code/profile_workloads.py`、`code/design_policies.py`，会得到新计时输入与新的独立实验。
本次最终测速在GPU0依次启动 `code/gpu_reservation.py`、`code/recover_speed.py`；运行间保留12GiB，模型初始化和计时前完整释放。质量启动 `code/orchestrate.py --role quality1/quality2` 分别绑定GPU4/7，GPU2/6启动 `code/quality_helper.py`，通过相同policy锁分担任务。不要同时在GPU0启动质量worker。
所有进程设置 PYTHONDONTWRITEBYTECODE=1、OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=1、HF_HUB_OFFLINE=1、HF_DATASETS_OFFLINE=1。原GPU3速度队列和GPU0质量worker只记录本次初始调度历史，最终复测使用上述GPU0速度与四质量worker即可。
`code/summarize.py` 只读取结果汇总；`code/finish_when_done.py` 等待测量结束后生成报告/图/审计。末尾执行 `code/recover_speed.py --cleanup-only` 释放全部已测可重建权重，检查结果后执行 `code/package_results.py` 封存。实际环境见 [execution.json](configs/execution.json)，原模型、canonical sparse states、wrapper 与 vLLM 路径见code/common.py及队列脚本。

已完成：251个速度组合（310组×7轮=2170轮）、202份质量文件（20200块，11886200计分token），267行完整比较表。全部正式计时GPU0，继承001/002的28155个文件及68个运行时源文件校验通过。

[最终报告](REPORT.md) · [完整CSV](results/summary_final.csv) · [审计](results/final_audit.json) · [图表](results/figures)。
