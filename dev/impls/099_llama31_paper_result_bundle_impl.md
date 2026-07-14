## 2026-07-14 - Consolidated paper result bundle
- 开发目的：把已完成的 Llama-3.1 prefill-only 与 prefill-decode 实验整理为可直接筛选论文点的全量表和数据集级 Pareto 图。
- 修改内容：新增聚合脚本，导出一张包含所有当前测得 ours 点、dense BF16 与全部 uniform 压缩方法的 Markdown/CSV 表；标注推荐的高质量、平衡和最高速候选。生成 ARC-Challenge、CNN/DM、DialogSum、IWSLT 四张图。
- 影响文件：`artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/make_paper_result_bundle.py`、`artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/`。
- 后续注意：PMPD 的 marlin-NVFP4 与 sparse-NVFP4 没有在 continuous closure 下补测，只保留已有 frozen legacy speed，并在表和图中以 `legacy*` 明确区分；不要用它们做小幅速度差比较。

## 2026-07-14 - Complete PMPD closure-point visibility
- 开发目的：避免总表只展示有下游任务分数的三个 mixed 点而遗漏已经完成真实速度/NLL closure 的策略。
- 修改内容：PMPD 总表现包含 point_000/002/004/006/008/009 六个实测 closure 点；point_006 与 point_008 明确标为“仅 closure、尚未测下游任务”。新增完整的 measured WikiText ΔNLL--speed 图，并以 off-scale 三角形标注 sparse-NVFP4 的 ΔNLL=113.6。
- 影响文件：`scripts/make_paper_result_bundle.py`、`artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/summary.md`、`all_measured_results.csv`、`pareto_prefill_decode_wikitext_nll.png`。
- 后续注意：三张任务图只能展示已有相应任务分数的 point_000/002/004/009；point_006/008 不应被误写为任务劣化，只是尚未执行该昂贵测试。

## 2026-07-14 - Single-GPU point_006 downstream expansion
- 开发目的：在仅 GPU 7 可用时，按用户要求逐策略补充尚未验证的 PMPD 中间 Pareto 点。
- 修改内容：导出 `point_006` checkpoint，并启动 GPU 7 专属的 persistent-vLLM 全量任务队列（CNN/DM-1000、DialogSum-1500、IWSLT-333；128-sample shard、batch 16）。
- 影响文件：`closure/checkpoints/point_006/`、`closure/tasks/point_006/`。
- 后续注意：任务仍在运行；完成后应依照 point_002/004 的相同 merger 与 metrics-only 流程生成分数，并重跑 paper bundle 脚本。

## 2026-07-14 - Point_006 task closure completed
- 开发目的：完成单卡队列的结果收口，并将该中高速点纳入最终任务曲线。
- 修改内容：GPU 7 完成三个数据集的 2833 条生成，随后完成标准 merge/metrics；point_006 在 CNN/DSum/IWSLT 上为 16.675 / 11.122 / 10.582。结果包更新为四个任务验证的 mixed 点（002/004/006/009），任务图新增 `ours-fast`。
- 影响文件：`closure/tasks/point_006/results/quality/`、`artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/`。
- 后续注意：point_008 仍是唯一尚未有三任务分数的已测 closure point；如果继续补点，应优先测试它。

## 2026-07-14 - Point_008 task closure completed
- 开发目的：完成最后一个尚未下游验证的 PMPD closure 策略点，使高速度区间也有真实生成任务支撑。
- 修改内容：GPU 7 完成 CNN/DM-1000、DialogSum-1500、IWSLT-333 的 persistent-vLLM 生成与标准 merge/metrics；point_008 的 CNN/DSum/IWSLT 分数为 16.274 / 8.957 / 10.546，对应 fresh continuous closure 的 1.585x 端到端加速。结果汇总与三张任务 Pareto 图均已更新，标为 `ours-faster`。
- 影响文件：`closure/checkpoints/point_008/`、`closure/tasks/point_008/results/quality/`、`scripts/make_paper_result_bundle.py`、`artifacts/exports/vllm/ours/llama3.1-8b-instruct/pareto_summary/`。
- 后续注意：六个 closure 点均已完成速度/NLL 测试，五个非 identity 的代表点完成三任务评测；若要进一步加密曲线，优先补测新的求解策略而非重复现有点。
