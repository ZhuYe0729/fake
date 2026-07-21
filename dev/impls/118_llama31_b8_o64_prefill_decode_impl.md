## 2026-07-19 - 057 canonical prefill-decode bootstrap
- 开发目的：为 Llama-3.1-8B-Instruct 建立与 056 Llama-2 实验同口径的独立 B=8、S=2048、O=64 prefill-decode 实验根目录。
- 修改内容：新增 `057_llama31_8b_instruct_b8_o64_canonical_pareto`；冻结 72 个带 hash 的两阶段策略及 100 条 2048+64 WikiText 标签；增加 Llama-3 canonical SparseGPT state 生成和 phase checkpoint 导出包装器；将已验证的 056 runner 参数化为可由显式环境变量复用。
- 正确性约束：`sparse_nvfp4` 增加 `--sparse-nvfp4-prequant-only`，其 state 仅含 SparseGPT 剪枝权重，最终 sparse-NVFP4 pack 只在 phase exporter 发生一次；不允许 `--prune` 回退。
- 影响文件：`artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/prepare_uniform_compressed.py`、056 shared runner 的路径参数化、057 新目录。
- 后续注意：canonical 状态完成后先运行真实 vLLM NLL（72 policy），再用同一 phase runtime 做速度校准/求解/独立进程测速，最后再测三项生成任务。磁盘不足以保留所有 checkpoint，必须逐策略 materialize→测量→清理临时 checkpoint。

## 2026-07-19 - NLL KV-cache OOM recovery
- 开发目的：保持 B=8 teacher-forced NLL 标签采集在混合 policy 下可完成。
- 修改内容：NLL completion 判定不再把 `gpu_memory_utilization` 当作质量标签身份的一部分；它只改变可分配 KV cache，不改变权重、策略、B=8、token 序列或 NLL 计算。已完成的结果可以保留，OOM policy 以更低 KV 配额重跑。
- 后续注意：该弹性只适用于 NLL/质量标签；正式速度测试仍固定 `gpu_memory_utilization=0.80`，不能混用。

## 2026-07-19 - Shared speed-runner path fix
- 开发目的：允许 057 使用共享 fresh-process runner，同时把产物写入自己的实验根目录。
- 修改内容：verifier 路径独立为 `COSPAQ_VERIFY_CHECKPOINT`，不再错误地从 057 根目录推导；Llama-3 环境显式指向经过验证的 056 verifier。
- 后续注意：speed checkpoint 仍是临时产物；每个成功速度点后删除，只保留 summary/iterations/log/provenance。

## 2026-07-19 - 057 model fits and Pareto solve
- 真实 vLLM NLL：72/72 policy 完成；混合 policy 在高 KV 配额发生 decode activation OOM 时，降 `gpu_memory_utilization` 重跑，不改变质量标签的权重、策略、B=8 或 token 口径。
- 质量模型：54/18 label-free coverage split 的 holdout MAE=0.0413、RMSE=0.0529、Spearman=0.9628。
- 速度模型：12-point fresh-process calibration 完成；monotone E2E calibrator 的 holdout MAE 从 499.16 ms 降到 224.24 ms。
- 求解：生成 12 个 B=8/O=64 phase-aware Pareto candidate；后续要对每点实测速、导出任务 checkpoint，并完成 CNN/DM、DialogSum、IWSLT 下游验证及报告。

## 2026-07-20 - Pareto fresh-process speed closure
- 开发目的：用统一的 phase runtime 对 12 个已求解的 Llama-3.1 B=8/O=64 Pareto policy 做端到端实测速度闭环。
- 修改内容：12/12 策略均以独立 vLLM 进程完成；成功后只保留 `summary.csv`、iteration/provenance 和日志，自动删除可复现 checkpoint。修正共享汇总脚本对旧 Llama-2 `b8o64000` 锚点的硬编码，使其也能从新布局中识别 `raw_speedup_vs_dense=1` 的全 BF16 锚点。
- 影响文件：`056_llama2_prefill_decode_b8_o64_canonical_pareto/scripts/summarize_pareto_speed.py`、`057.../speed/runs/point_*`。
- 后续注意：本阶段的 uniform 基线速度也已存在，但任务质量仍必须在相同 phase runtime 下逐策略重建 checkpoint 后评测。

## 2026-07-20 - Llama-3 generation-task runner compatibility
- 开发目的：使真实 vLLM 三生成数据集评测兼容 Llama-3.1 的 131072-token 默认 context 配置。
- 修改内容：任务 shard runner 接收 `MAX_MODEL_LEN`，057 显式使用 4096；否则旧 Llama-2 固定 `max_num_batched_tokens=15360` 会被新版 vLLM 以 `max_num_batched_tokens < max_model_len` 拒绝。IWSLT tokenizer 保持环境可覆盖，但 057 不覆盖 PMPD 规定的 Vicuna filter tokenizer。
- 影响文件：`035.../run_task_quality_shard.sh`、057 task materialization driver。
- 后续注意：4096 只改变 scheduler 的合法上下文上限，不改变 checkpoint、phase policy、解码采样或正式 B=8/O=64 速度口径；IWSLT 必须固定为历史 runner 使用的 Llama-2 filter tokenizer 所选 333 个样本，不能随被测模型 tokenizer 改变。新增 `--force-datasets`，用于数据协议改变时覆盖重跑，而不是只按相同行数错误复用旧 shard；此前 p00 的 IWSLT 临时结果会重跑且不纳入汇总。

## 2026-07-20 - Cross-policy task shard scheduler
- 开发目的：消除单 policy 分片尾部造成的 GPU 空置，同时不并行正式速度测试。
- 修改内容：新增 `run_batched_task_quality.py`。它先顺序物化有限个 policy checkpoint，再将这些 policy 的所有未完成 CNN/DM、DialogSum、IWSLT shard 放入同一 GPU 队列，按长 shard 优先调度；每张卡完成即领取任意下一分片。整批结束后删除所有可复现 checkpoint。
- 后续注意：首批限制为四个 checkpoint（约 60 GiB），在已释放的约 227 GiB 磁盘空间内保守运行；只用于下游任务精度，不改变速度测量的串行口径。

## 2026-07-20 - High-sparsity task-shard OOM recovery
- 开发目的：恢复最后一批高 sparse Pareto policy 的不完整生成分片。
- 观察：`point_010` 等 policy 在 `gpu_memory_utilization=0.75`、task batch=4 时出现 CUTLASS sparse-BF16 `cudaMalloc(matmul workspace)` OOM；成功分片没有错误且保留。
- 处理：恢复调度只选择行数不足的 shard，并降为 `gpu_memory_utilization=0.60`、batch=1，给 sparse workspace 留出确定 headroom。
- 正确性：这仅改变 vLLM KV 容量和请求分组；任务评测用同一 checkpoint、同一 prompt、温度 0 的贪心生成，质量指标不依赖 batch 分组。正式速度结果保持原 0.80 配额且不受影响。

## 2026-07-20 - Complete real-task validation and paper report
- 开发目的：完成 Llama-3.1 B=8/O=64 prefill-decode 的统一/异构策略下游验证并产出论文可用结果。
- 结果：5 个 uniform 与 12 个 Pareto policy 均完成 CNN/DM 1000、DialogSum 1500、IWSLT 333 条真实 phase-runtime 生成；51/51 metrics（ROUGE-L、BERTScore、SacreBLEU）完成。所有策略均有同口径 fresh-process 实测速度。
- 修改内容：新增最终报告生成器，产出宽 CSV、Markdown 总表及 CNN/DM ROUGE-L/BERTScore、DialogSum ROUGE-L/BERTScore、IWSLT BLEU 共 5 张 measured-speed Pareto 图。
- 关键产物：`057.../llama31_8b_instruct/task_quality/summary.csv`、`task_quality/report/summary.md`、`all_policy_task_results.csv`、`pareto_*.png`。
- 后续注意：`point_007` 的实测速率为 0.739x，明显低于同一 solver 筛选的预测，报告将其保留为数据点但不用于 ours 连线包络；推荐展示 `point_005`（1.262x，质量保持）与 `point_006`（1.321x，更强速度 trade-off）。
