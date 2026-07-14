## 2026-07-13 - Intermediate real-task evaluation launched
- 开发目的：验证 i34/i36/i37/i38 的 WikiText NLL trade-off 是否迁移到 CNN/DM、DialogSum、IWSLT 真实生成任务。
- 修改内容：新增 036 专用可恢复 runner，直接读取 intermediate checkpoint、复用连续 phase-hetero PMPD shard runner；启动 GPU 0–4 五卡任务并行，batch size=4、`.85` 显存、CNN/DM 1000、DialogSum 1500、IWSLT 333。
- 影响文件：`artifacts/debug/036_llama2_prefill_decode_intermediate_points/scripts/run_task_quality.py`、`task_quality_intermediate/`。
- 后续注意：速度汇总必须继续标注为剔除 >10 秒 phase-runtime stall 后的 screened median；生成任务指标不可用于重新拟合 WikiText 质量代理。

## 2026-07-13 - Continuous loading verification
- 验证：`pmpd_vllm_eval.py` 每个 shard 只构造一次 `LLM`；每 batch 后调用 `prepare_next_prefill()`，下一 batch 前调用 `wait_for_prefill_ready()`，不在每个 prefill→decode 或 batch 重载模型。
- 修改内容：修正 shard launcher 的过时注释，明确 vLLM 重启粒度为 shard，不是 batch。

## 2026-07-13 - Full intermediate task scores completed
- 验证：4×3 个策略/数据集结果完整，样本量严格为 CNN/DM=1000、DialogSum=1500、IWSLT=333，空输出均为 0；审计未发现重复 question-id。此前 22,664 行的误报来自批量 `wc` 同时累加每个文件和分组 total 行，实际生成记录为预期 11,332 条。
- 结果：i34/i36/i37/i38 在 screened 1.451/1.627/1.659/1.680x 下，CNN/DM ROUGE-L 为 23.752/23.833/23.843/23.754，DialogSum ROUGE-L 为 21.703/21.625/21.685/21.633，IWSLT SacreBLEU 为 18.304/18.949/18.639/19.029。

## 2026-07-13 - Task Pareto visualization
- 修改内容：合并 035 的旧策略、036 中间策略和 `.85` uniform baseline，输出三张真实任务 Pareto 图；中间点用绿色菱形并明确标为 `stall-screened`。
- 影响文件：`task_quality_intermediate/report/all_task_pareto_points.csv`、`pareto_cnn_dm_1000.png`、`pareto_dsum.png`、`pareto_IWSLT.png`。
