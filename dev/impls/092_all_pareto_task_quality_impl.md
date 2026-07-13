## 2026-07-12 - All-policy task-quality expansion started
- 开发目的：将已完成的 point 0、3、8、11 真实 PMPD 评测扩展到全部 12 个 prefill-decode Pareto 点，并绘制三张包含 baseline 的真实任务 Pareto 图。
- 现有数据：4 个代表点的连续 phase-hetero PMPD 结果完整，可作为可恢复任务池的一部分复用。

## 2026-07-12 - Remaining all-point PMPD evaluation launched
- 修改内容：连续 PMPD runner 和合并器支持 `--points`，可复用完整 shard；已启动 point 1、2、4、5、6、7、9、10 的三任务评测，point 0、3、8、11 直接复用。
- 运行配置：GPU 0、3、7，batch 4、`.85` 显存、每 shard 单次 vLLM 初始化并在 batch 间连续 phase 切换。
- 后续注意：生成完成后一次性做 12 点指标合并，并与 baseline 既有质量汇总合并作图。

## 2026-07-12 - Checkpoint lookup repair and relaunch
- 问题：初始 all-point runner 只使用 034 validation checkpoint，point 1、2、4、5、7、9、10 的完整 checkpoint 实际位于 035 calibration 目录，导致 tokenizer 路径解析失败。
- 修复：runner 现在优先选择包含 `model.safetensors` 和 tokenizer 配置的 035 checkpoint，再回退至 034 checkpoint。
- 运行：停止错误的无产出调度后，以原有 GPU 0、3、7 重启；完整 shard 会自动跳过，避免重复 point 0、3、8、11 的现有数据。

## 2026-07-12 - Long IWSLT shard observation
- 观察：point 1 的 IWSLT 连续 shard 在约 100 样本后长时间无日志，但最终由主任务池正常完成，属于极慢而非永久失败。
- 处理：曾启动 ≤100 样本 IWSLT recovery 作为保险；主池恢复进展后停止 recovery，避免与主池竞争 GPU 7 或产生重复正式输出。
- 当前状态：point 1 三任务完整；主池正在处理 point 2，其余策略继续排队。

## 2026-07-13 - All-policy real-task Pareto results
- 开发目的：完成全部 12 个 prefill-decode 策略的真实生成任务汇总，并在与 ours 相同 `.85` 速度口径下加入 uniform baseline。
- 修改内容：补齐 point 2、7、9、10 的恢复评测；基于已有完整分片严格校验并合并 12×3 条任务结果。由于 BERTScore 在该节点未能稳定使用 CUDA，最终绘图采用评测套件同实现的逐样本 ROUGE-L（CNN/DM、DialogSum）和 SacreBLEU（IWSLT）；未在最终汇总中填入新的 BERTScore。
- 影响文件：`artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/task_quality_all/summary.csv`、`report/all_task_pareto_points.csv` 及三张 `pareto_*.png`。新增轻量指标合并器，并修复 `combine_task_quality.py` 缺少的 `main()` 入口。
- 验证：36/36 个 `(point, dataset)` 条目完整，样本量分别为 CNN/DM 1000、DialogSum 1500、IWSLT 333；baseline 速度已重新以 `.85` 协议测量。point 9 的速度波动仍在图中标记为不稳定且不参与 ours frontier。
