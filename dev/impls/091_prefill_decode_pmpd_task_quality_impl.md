## 2026-07-12 - Task-quality implementation started
- 开发目的：验证 `.85` 实测速度与 WikiText Pareto 趋势是否迁移到真实 PMPD 生成任务。
- 范围：point 0、3、8、11，覆盖 CNN/DM-1000、DialogSum-1500、IWSLT-333；使用独立进程 phase-hetero vLLM batch runner。
- 产物位置：`artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/task_quality/`。

## 2026-07-12 - Isolated PMPD runner launched
- 修改内容：新增可断点续跑的 shard runner、GPU-pool 调度器和严格合并/指标汇总脚本；每个 batch 以 fresh-process phase-hetero vLLM 执行，固定 `.85` 显存配置。
- 验证：shell 语法、Python 编译和 CLI 参数检查通过；已为 point 0 的 CNN/DM 三个 shard 在 GPU 0、3、7 启动首批任务。
- 运行约束：其余 GPU 当前有外部负载，未强行复用，调度器只使用空闲卡；重启同一命令会跳过完整 shard 并继续。

## 2026-07-12 - Switched to continuous phase-hetero PMPD execution
- 发现：vLLM 的既有 `pmpd_vllm_eval.py` 已支持单次 `LLM` 初始化、batch 间 `prepare_next_prefill()` 与 `wait_for_prefill_ready()`；此前 runner 每 batch 重启 vLLM 是不必要且极慢的。
- 修改内容：shard runner 改为每 shard 调用一次该连续 evaluator；结果改写入独立 `task_quality_continuous/`，避免与旧 fresh-process partial shards 混用。
- 运行状态：已终止旧 fresh-process pool，确认 GPU 0、3、7 释放后启动连续 phase-hetero pool；首批为 point 0 CNN/DM 三个 shard。

## 2026-07-12 - PMPD task-quality validation completed
- 验证结果：4 个策略 × 3 个任务的 12 个 shard 集均完整合并，样本数严格为 CNN/DM 1000、DialogSum 1500、IWSLT 333，全部 `empty_predictions=0`。
- 结果产物：`task_quality_continuous/summary.csv` 与 `summary.md`，包含 `.85` 实测 E2E/speedup、WikiText ΔNLL 和 ROUGE-L、BERTScore、SacreBLEU。
- 结论：point 3 在低损失区保持或改善 CNN/DM 与 IWSLT，同时 DialogSum 仅轻微下降；point 8 和 max-speed point 11 在 CNN/DM、DialogSum 上保持接近 dense 的质量，但 IWSLT 的 BLEU 随压缩增大而下降，支持存在任务依赖的真实质量折中。
