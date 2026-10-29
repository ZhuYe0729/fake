## 2026-10-27 - 冻结新点真实mixed调度测试
- 建立独立047，沿用040固定九map、006两个chunk1024代表case，复用修正后的算子与逐行phase路由；旧006/040/主表未修改。
- 新64窗口实际case teacher forcing、BF16 routed/no-op门槛、三轮首尾BF16及全部原始captures/路由/显存诊断均已落实；失败重试用独立attempt目录，保留此前captures。
- CPU检查已通过全部Python语法、64×4160输入/分组、九map来源SHA及明确的mixed row索引fixture；尚未声称GPU实验通过。
- 控制器会话7288已启动，等待044/045/046完成，避免挤占本夜优先证据。完成后交付reviewer2/qwen_updated_mixed_run047完整18行表。

## 2026-10-27 - 核心Qwen重复先于mixed扩展
- 发现需核实旧eager/当前Graph参考差异，047顺延到048统一完整生成/速度重复之后，样本、九map、case和评分规则均未更改。
- 仅重启无子任务的自有等待器，全部旧日志保留；OMP4/MAX_JOBS4/offline环境在首次GPU工作前恢复，重排凭证位于night161/reference。
