# Run003 执行记录

最终正式速度统一为GPU0；下列初始调度的GPU3部分记录完整保留，卡切换说明见文末。

- 工作负载、预算、对照和100个测试块在测量前冻结，见 configs/design.json 与 data/test_samples.json。
- profile_workloads.py：360个实际shape/格式完整forward成本；普通INT8融合前后24项数值检查。初次阈值过严失败及源码保留在 reference/diagnostics，正式数值容差与002一致。
- BF16最大负载试跑通过。首次请求前阶段恢复调用已修正，首次smoke失败日志保留。统一6GiB KV容纳最大实际并发token数。
- design_policies.py：267个比较项/47个唯一method-map；DASO两个预算，HAWQ相应Linear延迟上限。独立audit_allocation.py重放逐字节一致，CPU重放约4.9s。
- 正式速度仅物理GPU3，依次运行同一策略的全部指定负载；质量在速度完成后通过共享policy锁领取，GPU0/4/7，后加空闲GPU2/6。
- 每次子进程调用保存命令、全部实验Python源码哈希、代码快照、日志和退出记录；失败日志不覆盖。旧001/002只读，PYTHONDONTWRITEBYTECODE=1。
- finish_when_done.py 在六个测量worker均完成后生成汇总、报告、图与完整审计；后续检查图表再封存manifest和reviewer证据包。
- 临时导出位于 /dev/shm/daso_rebuttal_003_20261025，只在全部指定速度和质量任务成功后释放可重建的model.safetensors；所有JSON metadata、日志、代码、结果和文档保留。

## 外部占卡后迁移
- GPU3在新模型启动时被外部20.4GiB任务占用，vLLM在初始化前拒绝继续；用户明确回复使用空闲卡即可。
- 原210个策略/负载计时组合移至results/diagnostics/speed_gpu3_partial；固定策略、数据与质量结果保留，所有最终速度在GPU0完整重测，不跨卡拼接。
- 空闲quality0进程交接给其余四个质量worker；GPU0运行gpu_reservation.py与recover_speed.py。12GiB占位仅在导出及模型切换期间保留，LLM初始化和计时前释放，并有nonce握手记录。
- 最终速度缓存位于/dev/shm/daso_rebuttal_003_gpu0_20261025；可硬链接旧缓存避免无谓重建，删除旧链接不会影响在用的新链接。所有速度/质量完成后再释放模型二进制。
- 完整协议修订见configs/measurement_card_change.json；最终CPU报告/审计等待GPU0速度和全部质量worker完成。

2026-10-25: final GPU0 speed, reservation helper and CPU finalizer exited 0. All quality workers completed; original idle GPU0 quality worker retired intentionally (-15). Full audit passed. Final report reviewed, figures inspected, owned temporary exports released; ready for manifest and reviewer packages.
