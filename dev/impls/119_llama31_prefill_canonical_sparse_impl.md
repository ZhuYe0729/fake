# 119 Llama3.1 canonical prefill-only implementation

## 2026-07-20 - Experiment scaffold and reusable-asset contract
- 开发目的：建立 Llama3.1 prefill-only 的 canonical 闭环实验入口，避免重做 057 已验证的权重/局部误差资产。
- 修改内容：新增 058 README、场景常量、bootstrap/asset verification、速度 anchor、真实 vLLM NLL、质量拟合与求解脚本。
- 影响文件：`artifacts/debug/058_llama31_prefill_canonical_sparse_quality_recalibration/`。
- 后续注意：057 canonical state 只能引用且必须以 hash/provenance 验证；任何 sparse exporter 不得使用 `--prune`。

## 2026-07-20 - Speed/NLL calibration and closure split
- 开发目的：完成速度与纯 prefill 质量代理的可复现实测，并避免临时磁盘与测速并发污染。
- 修改内容：速度临时 checkpoint 改至 `/root/wja` 实验文件系统；修复新版 vLLM 的 `max_num_batched_tokens >= max_model_len` 要求；增加日志化、可恢复的 NLL 调度器；修复 solver 中非 BF16 action 的零质量 bin；将 closure 拆为可多卡并行的 NLL 和单卡隔离的五次速度。
- 验证结果：057 canonical state hash/feature coverage 通过；速度校准 holdout MAE 从 162.46 ms 降至 33.40 ms；72 个 canonical phase-vLLM prefill NLL 标签完成；质量模型 holdout MAE 0.02619、RMSE 0.03305、Spearman 0.99174。
- 后续注意：当前 10 个 selected mixed 点的 closure NLL 已完成，GPU 1 正在串行完成 speed closure；下游任务只能使用闭环后的实测速率/NLL 选择点。

## 2026-07-20 - Dense-NVFP4 bridge closure
- 开发目的：补齐原求解器在 dense-NVFP4 邻域从 pure dense 策略跳至 sparse 策略造成的曲线空洞，并用真实 phase-vLLM 数据验证。
- 修改内容：按每个模块的预测质量代价/预测速度收益排序，生成 72/88/104/120 个 dense-NVFP4 模块的 BF16+dense-NVFP4 过渡策略；closure NLL/speed scheduler 支持显式策略列表；closure 汇总包含 bridge 行。
- 验证结果：四个 bridge 的真实 NLL 与单卡独立五次 E2E 速度均完成。120-module bridge 为 `1.836x / ΔNLL=0.09311`，紧邻 uniform dense-NVFP4 的 `1.849x / ΔNLL=0.09650`，且质量略好；其余 bridge 提供 `1.434x/0.05593`、`1.509x/0.07024`、`1.743x/0.08091` 的连续过渡。
- 后续注意：bridge 速度必须继续与 uniform anchor 相同的 phase-vLLM benchmark 口径串行执行，不得并发测速。

## 2026-07-20 - Paper-facing prefill task validation
- 开发目的：将已闭环的实测 NLL/速度曲线，验证到真实 prefill-only 下游任务，形成论文图表所需的最终质量数据。
- 修改内容：新增可恢复的多 GPU task scheduler；选择 5 个 uniform、3 个低损失 mixed、4 个 dense-NVFP4 bridge 与 3 个高加速 mixed，共 15 个代表点。每点在同一 phase-vLLM runtime 上运行 WikiText、WinoGrande、ARC-Easy、ARC-Challenge 和 MMLU。
- 影响文件：`058/.../scripts/run_task_selection.py`，以及 `058/.../llama31_8b_instruct/task_quality/`。
- 后续注意：任务评测不计入速度；最终图横轴只读取已独立完成的五次 E2E closure 测速中位数。

## 2026-07-20 - Task recovery and final report
- 开发目的：处理 MMLU 的显存/缓存元数据不稳定，并把闭环速度、NLL 和五项真实任务结果汇总为论文产物。
- 修改内容：task runner 增加保守显存模式、HF 离线缓存模式和每策略锁；调度器修复 `point_*` 路径判别与单点失败不中断队列；新增 `build_task_report.py`。
- 验证结果：15/15 策略的 WikiText、WinoGrande、ARC-Easy、ARC-Challenge、MMLU 均完成。产物生成于 `058/.../task_quality/report/`；bridge-120 为 `1.835x / ΔNLL=0.0931`，ARC-Challenge `0.5384`，优于 uniform dense-NVFP4 的 `1.849x / 0.0965 / 0.5111`。
- 后续注意：下游任务统一采用离线缓存和保守显存，不影响独立、五次中位数的速度协议。
