## 2026-07-13 - Plan initialized and 034 audit started
- 开发目的：以 prefill-decode 已验证的方法重新建立 Llama2-7B prefill-only 帕累托求解与验证流程。
- 修改内容：建立 plan 096；确认 034 仅有五个真实 prefill-only 策略点，选点依赖 raw kernel latency sum，尚无独立 E2E 校正、全曲线实际 NLL 和任务级验证。
- 影响文件：`dev/plans/096_llama2_prefill_only_pareto_plan.md`。
- 后续注意：所有新实验写入新的 `artifacts/debug/` 根目录，GPU 仅使用 0--4，且不能混用 prefill baseline runner 口径。

## 2026-07-13 - Prefill-only E2E calibration sweep launched
- 开发目的：填补 034 仅有 5 个实测点、无法独立验证 policy-level E2E latency 校正的问题。
- 修改内容：新建 `artifacts/debug/037_llama2_prefill_only_pareto/`；冻结 point 1/3/6/9/11/13/15 作为新增速度校准点，point 0/4/8/12/16 保留为初始 held-out 检查；新增按点导出 checkpoint 并执行 1 warmup + 5 fresh-process prefill benchmark 的可恢复脚本。
- 验证：脚本通过 `bash -n`，7 个 policy JSON 均存在；导出/测速已在 GPU 0--3 启动。首次调用因脚本无执行位未实际进入 GPU，已改为显式 `bash` 调用；后续进程检查确认四个 exporter 正在运行。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/README.md`、`calibration_points.txt`、`scripts/run_calibration_point.sh`。
- 后续注意：等待 checkpoint/测量完成后先进行 held-out E2E 校正误差评估；不将 raw kernel sum 直接用于最终前沿。

## 2026-07-13 - Calibration runner environment correction
- 开发目的：修复 speed stage 未实际使用 vLLM conda environment 的启动错误。
- 修改内容：checkpoint exporter 保持 `cospaq` 环境，vLLM benchmark 改为显式使用 `vllm` 环境。
- 验证：point 1/3/6/9 checkpoint 已成功导出；它们原来的 warmup 日志均为 `ModuleNotFoundError: pydantic`，定位到 `cospaq` 环境，而非模型或策略错误。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_calibration_point.sh`。
- 后续注意：重启时会复用已完成 checkpoint，只补 vLLM speed run 和尚未导出的 point 11/13/15。

## 2026-07-13 - E2E speed calibration completed and validated
- 开发目的：量化 raw kernel latency sum 到真实 prefill-only vLLM E2E latency 的校正收益。
- 修改内容：完成 point 1/3/6/9/11/13/15 的 checkpoint 和各 5 次 fresh-process 测速；新增单调 policy-level E2E 校正与 strict held-out / all-point LOO 验证脚本。
- 验证：新 7 点均为 5/5 有效样本。对 034 保留点 0/4/8/12/16，dense-anchor raw 预测 MAE 为 62.56 ms，单调校正降至 11.31 ms；12 点 LOO MAE 从 71.67 ms 降至 10.78 ms。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/measurements/`、`checkpoints/`、`scripts/fit_e2e_calibrator.py`、`e2e_calibration.csv`、`e2e_calibration_metrics.json`。
- 后续注意：校正模型已经通过速度验证，但尚未写入 Pareto solver；下一步检查/复用 prefill-only WikiText 质量模型并在 corrected speed objective 下重求策略。

## 2026-07-13 - Corrected candidate curve and actual-NLL validation launched
- 开发目的：把已验证的 E2E 速度校正接入 prefill-only Pareto 候选，并将预测纵轴替换为真实 WikiText NLL。
- 修改内容：确认使用 033 的 normalized-pooled quality proxy（72 个受控策略，54/18 固定切分，holdout MAE 0.126、Spearman 0.774）；生成 17 点 corrected candidate curve。由于 E2E calibration 是 raw latency 的单调映射，固定质量预算下 DP 最小 raw latency 与最小 corrected latency 的策略相同，因此保留原离散策略、只修正速度轴。
- 验证：shell 静态检查通过；启动 point 1/3/6/9/11/13/15 的固定 100-block prefill NLL，分配到 GPU 0--4，物理 GPU id 直传以避免 `CUDA_VISIBLE_DEVICES` 重映射错误。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/pareto/corrected_candidate_curve.csv`、`pareto/corrected_candidate_metadata.json`、`scripts/build_corrected_candidate_curve.py`、`scripts/run_actual_nll_point.sh`。
- 后续注意：NLL 完成后与 034 的 point 0/4/8/12/16 合并，按两个实际轴重新进行非支配筛选；任务级生成评测只对筛选后的代表点执行。

## 2026-07-13 - Measured prefill-only NLL frontier completed
- 开发目的：以真实 E2E speed 和真实 WikiText NLL 替换 candidate surrogate 坐标，并得到可用于任务级验证的前沿。
- 修改内容：7 个新策略 NLL 全部完成且无 OOM/评测错误；新增合并、数值噪声零下限（`|ΔNLL|<1e-3`）和非支配筛选脚本，输出汇总 CSV、frontier CSV 和图。
- 验证：12 个 ours 点均有 5 次 speed median 与 100-block NLL。实测 front 上 ours point 4/6/8/9 填补 dense 到 dense-NVFP4 的低损失区间；point 15/16 则填补 dense-NVFP4 到 sparse-NVFP4 的高加速区间。point 11--13 等被 uniform dense-NVFP4 支配，未进入 frontier。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/actual_nll/`、`scripts/build_actual_nll_frontier.py`、`report/actual_nll_speed_summary.csv`、`report/actual_nll_measured_frontier.csv`、`report/pareto_speedup_vs_actual_wikitext_nll.png`。
- 后续注意：下一步只对实测非支配的代表策略做 CNN/DM、DialogSum、IWSLT 任务级验证；不对已被支配点浪费评测资源。

## 2026-07-13 - Representative generation-task evaluation launched
- 开发目的：验证实测 WikiText Pareto 趋势能否迁移到真实生成任务。
- 修改内容：选择已完成 checkpoint 的 point 6/9/15（低损失、中段、高加速），复用 PMPD phase-heterogeneous shard runner；先完成 point 6 IWSLT 100-example smoke shard，随后在 GPU 0--4 启动完整 CNN/DM、DialogSum、IWSLT 评测。runner 在每 shard 内只加载一次模型并复用 batch；已完成 shard 可自动跳过。
- 验证：smoke shard 100/100 JSONL 输出完整，无空输出或 phase-switch error；GPU 0--4 占用确认完整 launcher 已实际开始执行。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_task_quality.py`、`task_quality/`。
- 后续注意：完整任务预计约 1.5--2 小时；完成后合并 metrics、审计 question coverage，并生成三张 task-specific Pareto 图。

## 2026-07-13 - Recoverable physical-GPU task runner correction
- 开发目的：修复并行 shard 运行时 `CUDA_VISIBLE_DEVICES` 在受控启动链中未稳定传递、导致多个 vLLM worker 竞争同卡并在 `.9` KV 预留检查失败的问题。
- 修改内容：新增在 vLLM Python 进程内显式 `torch.cuda.set_device(physical_gpu)` 的 wrapper 与 shard runner；task quality launcher 改为传递物理 GPU id，并把仅影响 KV 容量而不影响确定性任务分数的 `gpu_memory_utilization` 改为 `.8`。恢复 launcher 仅使用当前空闲的 GPU 1/2/4，自动跳过已完成 shard。
- 验证：已完成 7 个完整 shard 均保留；失败日志确认根因是 free memory 低于 `.9` 预留，不是模型/数据错误；修复后的 recovery launcher 已开始报告已完成 shard。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/run_pmpd_on_gpu.py`、`scripts/run_task_quality_shard.sh`、`scripts/run_task_quality.py`、`task_quality/logs/recovery_launcher.log`。
- 后续注意：任务运行中应避免再次以旧 runner 启动；完成后需审计所有 shard 的行数，不把失败 shard 当作空结果。

## 2026-07-13 - Task-quality watchdog enabled
- 开发目的：避免 launcher 因单 shard 失败退出后，GPU 1/2/4 长时间空闲而无人恢复。
- 修改内容：新增每 60 秒检查一次完成 shard 数的轻量 watchdog；若尚未达到 36 个完整 shard 且指定恢复 launcher 不存在，则按修正后的 physical-GPU/.75 配置重启；达到 36 后自动退出。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/watch_task_quality.sh`、`task_quality/logs/watchdog.log`。
- 后续注意：watchdog 只负责恢复生成 shard，不合并 metrics；完成后仍需显式审计行数、计算 metrics 和出图。

## 2026-07-13 - Generation-task metrics and Pareto figures completed
- 开发目的：完成 prefill-only Pareto 的真实生成任务迁移验证。
- 修改内容：生成阶段完成 36/36 shard；发现并修复一个早期并发重试造成的 point 6 CNN/DM JSONL 损坏 shard，升级完成判定为逐行 JSON 校验并仅重跑该 shard。随后合并去重 JSONL、计算 9 组 PMPD metrics，并生成三张 task-specific Pareto 图。
- 验证：9/9 metrics 文件写出；point 6、9 在全部三个数据集均无空输出。point 6 在 1.074x 下维持 CNN/DM 24.114、DialogSum 21.690、IWSLT BLEU 21.141；point 9 在 1.363x 下仍保持 CNN/DM 24.089、DialogSum 21.489、BLEU 19.264。高加速 point 15 任务退化严重（CNN/DM 171 个空输出），应在任务图中保留为负向证据而不作为主质量前沿点。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/task_quality/summary.csv`、`task_quality/results/`、`task_quality/report/`、`scripts/merge_and_plot_task_quality.py`。
- 后续注意：若需补充任务级高速度有效点，应从 NLL 前沿的 point 12--14 中单独筛选；不要把 point 15 的 WikiText 中等 ΔNLL 直接外推为稳定生成质量。

## 2026-07-13 - ARC-Challenge prefill-only validation added
- 开发目的：用不依赖长自回归 decoding 的多选题条件似然评测，直接验证 prefill-only 策略的下游质量趋势。
- 修改内容：新增 ARC runner，复用 WikiText NLL 使用的同一份 prepared 权重安装逻辑；筛查集合包含五个统一压缩基线和 ours point 6/9/15，速度轴仍采用已实测的 vLLM b=8、input=2048 加速比。
- 验证：`lm_eval 0.4.12` 在 `cospaq` 环境可用；评测脚本对策略仅安装 `prefill_method`，不会混入 decoding 权重。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/scripts/evaluate_arc_challenge.py`、`scripts/run_arc_screen.sh`。
- 后续注意：先以固定 128 道题确认全体趋势与运行稳定性，再对实际前沿候选执行完整 ARC-Challenge。

## 2026-07-13 - ARC-Challenge full prefill-only validation completed
- 开发目的：验证 prefill-only 求解的 WikiText-NLL 趋势能否迁移到不依赖 decoding 的真实多选任务。
- 修改内容：完成五个 uniform baseline 与 ours point 6/9/15 的完整 ARC-Challenge（1,172 题、0-shot、条件似然）评测；新增将任务分数和既有 vLLM b=8/input=2048 实测速度合并并进行非支配筛选的报告脚本。
- 验证：point 6/9 的 `acc_norm` 为 0.4317/0.4309，接近 dense BF16 的 0.4334，且高于 uniform dense-NVFP4 的 0.4283；point 15 为 0.3234，成为高加速端的显式质量 trade-off。全部指标来自完整 1,172 题而非筛查子集。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/arc_challenge/`、`scripts/build_arc_challenge_report.py`。
- 后续注意：该结果只声明 prefill-only task quality；不能替代 prefill-decode 的生成稳定性评测。

## 2026-07-13 - Dense-NVFP4 bridge refinement launched
- 开发目的：加密 uniform dense-NVFP4 以上的速度区间，寻找可用于论文表格的“更快且质量接近”混合策略。
- 修改内容：以同一冻结质量代理和 3,200 DP bins 重求 102 个候选；新增可恢复的 refined-policy checkpoint/E2E runner 与 100-block WikiText NLL runner，计划先实测四个预测 1.87--1.95x 的桥接候选。
- 验证：加密求解成功生成 102 个离散策略；速度阈值附近候选的预测质量成本显著高于 uniform dense-NVFP4，因此必须以真实 NLL/ARC 排除代理误判。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/refined_solver/`、`scripts/run_refined_bridge_point.sh`、`scripts/run_refined_bridge_nll.sh`。
- 后续注意：若四点的实测 NLL 均远离 dense-NVFP4，应报告当前动作空间内不存在该目标，而非挑选偶然噪声点。

## 2026-07-13 - Paper-ready prefill-only ARC frontier completed
- 开发目的：形成不回避 uniform baseline、且两轴均为实测的 prefill-only 论文图。
- 修改内容：补齐真实 NLL 非支配 mixed 集合中 point 4/8/16 的完整 ARC-Challenge；主图合并五个 uniform baseline 与 point 4/6/8/9/15/16，另输出高质量区间放大图以展示细粒度折中。
- 验证：point 8 在 1.213x 达到 ARC `acc_norm=0.4334`，与 dense BF16 持平；point 9 在 1.363x 仍为 0.4309。最终非支配曲线由 point 8/9、dense-NVFP4、point 15/16、sparse-NVFP4 构成，统一方法作为同一候选集参与筛选。
- 影响文件：`arc_challenge/full/ours_point_004.json`、`ours_point_008.json`、`ours_point_016.json`、`arc_challenge/report/`、`scripts/build_arc_challenge_report.py`。
- 后续注意：ARC 全量得分的标准误约 0.014；应将 point 8 相对 dense 的微小差异表述为“within measurement uncertainty / matched quality”，而非绝对优于 dense。

## 2026-07-13 - ARC intermediate trade-off densified
- 开发目的：填补 1.36x--1.87x 的中速区，使论文图展示连续而非只含少数端点的实测 trade-off。
- 修改内容：完成 point 11/12/13 的完整 ARC-Challenge；绘图保留所有实测点、以非支配并集连线，并新增覆盖 1.0x--1.9x 高质量平台的放大图。
- 验证：point 11/12/13 的 ARC `acc_norm` 分别为 0.4403/0.4420/0.4343，对应实测速度 1.635x/1.726x/1.811x；point 12/13 连同 uniform dense-NVFP4 构成实际 ARC 中段前沿。
- 影响文件：`arc_challenge/full/ours_point_011.json`、`ours_point_012.json`、`ours_point_013.json`、`scripts/build_arc_challenge_report.py`、`arc_challenge/report/`。
- 后续注意：point 14 的真实 speed/NLL 尚未补齐；其 GPU 4 启动请求因审批通道中断，未以插值值混入实测图。

## 2026-07-13 - Two-scenario paper summary assembled
- 开发目的：为论文汇总 Llama2 两个 serving 场景的实测速度和任务质量，并选择可解释的 ours 代表点。
- 修改内容：新增跨场景 Markdown 表；prefill-only 选择实际 ARC 并集前沿的 point 12，prefill-decode 选择 1.714x 且三任务接近 dense 的 point 11；保留全部 uniform 参考及来源路径。
- 影响文件：`artifacts/debug/037_llama2_prefill_only_pareto/report/llama2_two_scenario_paper_summary.md`。
- 后续注意：表中速度仅可在各自场景内比较；prefill-decode 的 baseline 数值来自与 task report 对齐的 comparison CSV，不能与另一 runner 的旧测速混用。
