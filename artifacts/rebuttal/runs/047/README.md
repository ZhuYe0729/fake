# Qwen新点：真实chunked/mixed调度

固定040同九map、006两代表case，先64窗口实际mixed teacher forcing与BF16 no-op校验，再同卡三轮2warmup+5timed burst。双表示驻留变体单列内存，不混入主表，不称open-loop生产服务。等待044/045/046完成后开始。

[设计](configs/design.json) · [计划170](../../../../dev/plans/170_updated_qwen_mixed_scheduler_plan.md)
