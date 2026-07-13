## 2026-07-11 - NLL Pareto modeling v1 scaffold and validation launch
- 开发目的：在真正求解帕累托界限前，为 Llama2-7B-Chat vLLM 建立并验证速度预测与 NLL 精度预测输入。
- 修改内容：新增 30 个固定的分阶段异构校准策略（21 train / 9 holdout）、PMPD 三数据集各 100 条样本、教师强制 NLL 采集、正性 local-error 聚合拟合、原始 KernelLatencyPredictor 聚合、均匀基线速度验证，以及 8 卡独立 NLL shard 调度/合并脚本。
- 影响文件：`artifacts/exports/vllm/ours/llama2-7b-chat/pareto/nll_modeling_v1/`、`dev/plans/085_llama2_7b_chat_vllm_pareto_nll_modeling_plan.md`。
- 验证：Python 静态编译、样本/策略生成、60 个原始速度预测和已有均匀 vLLM 基线对比已完成；完整 60-policy NLL 标定已启动，结果合并后再拟合并报告 holdout 指标。
- 后续注意：decode 的 sparse-NVFP4 对这些 M=16 fused shape 没有可用 kernel，已从 decode 动作空间排除；帕累托优化器仍保持 TODO。

## 2026-07-11 - GPU worker isolation correction
- 开发目的：让 30 个 7B NLL worker 可以安全地利用 8 卡并行。
- 修改内容：发现 elevated 子进程不可靠地重映射 `CUDA_VISIBLE_DEVICES`，改为直接传物理 GPU id；又限制 launcher 同时最多一个 worker/GPU，防止四个 7B 模型同时落到同一张 32GB 卡。
- 影响文件：`scripts/run_parallel_nll.sh`。
- 后续注意：第一次并行尝试的 OOM worker 已停止，尚未产生有效 shard；修正后可重新启动完整标定。

## 2026-07-11 - 可恢复的缺失策略补测器
- 开发目的：只补测 OOM 后缺失的 NLL 结果，并保证每张物理 GPU 同时只有一个 7B 进程。
- 修改内容：新增 `run_missing_nll.sh`；按 GPU lane 串行分配 policy，检测并跳过已有 shard，任一 lane 失败则保留已有结果供下次恢复。
- 影响文件：`scripts/run_missing_nll.sh`。

## 2026-07-11 - 全量 NLL 校准与拟合完成
- 开发目的：完成两个场景的策略 NLL 采集并量化 v1 精度模型泛化能力。
- 修改内容：成功恢复并完成两个场景各 30 个 shard，自动合并 NLL、拟合 21/9 train-holdout 模型并产出预测表。
- 验证：prefill-only holdout Spearman 为 0.667；prefill-decode holdout Spearman 为 -0.483，表明当前将旧的未融合 local error 直接用于带 80 倍 decode 权重的 v1 模型不可用，需要在后续精度建模中重新校准 decode 特征/尺度后才可进入帕累托求解。
