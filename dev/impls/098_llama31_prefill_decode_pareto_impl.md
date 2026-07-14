## 2026-07-13 - Phase-aware experiment initialization
- 开发目的：建立 Llama-3.1-8B-Instruct prefill-decode Pareto 的独立、可复现实验根目录。
- 修改内容：冻结 b=16 / in=2048 / out=80 的双相 action support；生成固定 WikiText 样本和 72 个（54 train / 18 holdout）不同 prefill/decode 分配的策略；加入 phase-NLL、phase-local-error、速度校准脚本。
- 影响文件：`artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/`，`dev/plans/098_llama31_prefill_decode_pareto_plan.md`。
- 后续注意：质量标签采用 `ΔNLL_prefill + 80·ΔNLL_decode`；导出 checkpoint 的并发必须保持在 4 个以内以避免耗尽共享磁盘。

## 2026-07-13 - 质量测量启动与路径修正
- 开发目的：启动质量代理所需的局部误差与 WikiText 标签测量。
- 修改内容：修正生成器将 phase 策略写入错误 `prefill_only` 子目录的问题，重新生成并验证 128 个模块存在 phase 异构分配；启动 phase-local 误差和 p00/p01/p02 NLL 测量。
- 影响文件：`generate_inputs.py`、`logs/`、`local_errors/`、`nll_shards/`。
- 后续注意：p00 是共享 dense-reference，必须完成后才可让其余 shard 复用基线；此前误发到同 GPU 的诊断任务已确认仅为显存竞争，未写入或污染结果。

## 2026-07-13 - 锚点修复与并行建模测量
- 开发目的：保证 uniform 对照在两个阶段使用完全相同的方法，并提高 GPU 利用率。
- 修改内容：固定 p00--p04 为严格两阶段 uniform；对错误 p00 标签启动覆盖重测；GPU 5--7 运行其余 NLL shards，GPU 2--4 并行导出 p00--p02 checkpoint 并测连续 phase E2E；新增基于冻结质量模型和双相 kernel latency 的 MCKP screening solver。
- 影响文件：`generate_inputs.py`、`solve_predicted_pareto.py`、`nll_shards/`、`speed_calibration/`。
- 后续注意：NLL shard 单策略耗时为分钟级，预计三卡完整标签仍需约一小时；checkpoint 已占约 30 GB，后续速度校准须在完成测量后回收临时 checkpoint 或保持不超过四个并发。

## 2026-07-13 - 连续 phase runner 可行性校正
- 开发目的：让所有混合策略的速度校准在相同、可执行的 vLLM 内存协议下完成。
- 修改内容：诊断发现 `.9` utilization 下 dense-NVFP4 phase activation 缓冲 OOM；新增独立 `speed_calibration_util085` 测量组，统一固定 `.85`，此前 `.9` p00--p02 数据仅保留作协议诊断且不参与拟合；已释放完成 p00 的临时 checkpoint。
- 影响文件：`run_speed_calibration_point.sh`、`fit_speed_calibrator.py`、`speed_calibration_util085/`。
- 后续注意：最终图表将标注 mixed-policy continuous-phase `.85` 协议，并且只与同协议的 dense anchor 比较；现有 uniform `.9` 基线保持冻结、不被修改。

## 2026-07-13 - NVFP4 phase initialization observation
- 开发目的：验证 `.85` 协议能否稳定覆盖 dense-NVFP4 phase 策略并控制校准墙钟时间。
- 修改内容：p01 成功完成 warmup 与重复测量；记录每个 fresh vLLM process 的 NVFP4 phase initialization 约 118 秒，随后将 p02/p03 分配到 GPU 3/4 并行测量。
- 影响文件：`speed_calibration_util085/runs/`、`logs/speed_speed_calibration_util085_*`。
- 后续注意：该初始化时间不计入 generate-only E2E latency，但会影响实验时长；结果汇总时只使用 `elapsed_ms`，不把加载时间混入速度指标。

## 2026-07-13 - NLL completion dependency gate
- 开发目的：避免长时间 NLL shard 队列结束后依赖步骤未被触发。
- 修改内容：新增并后台启动 `advance_after_nll.sh`；它严格等待 72/72 shard 后依次执行合并、质量代理拟合、速度校准设计和双相 Pareto screening 求解。
- 影响文件：`scripts/advance_after_nll.sh`、`logs/advance_after_nll.log`。
- 后续注意：脚本仅在所有固定策略标签就绪后执行，因此不会把部分数据或中间结果写入最终模型。

## 2026-07-13 - Mixed calibration phase-pair legality fix
- 开发目的：消除 mixed policy 在 phase runtime profile 初始化后无法继续的情况。
- 修改内容：确认随机质量校准策略含有 runtime 不支持的 phase method pairs；新增合法投影（同方法或 dense-NVFP4↔W4A16），p37/p39 分别修复 74/94 个模块，删除未产生测量的无效 checkpoint 后重新导出；速度设计改为读取该合法 policy。
- 影响文件：`make_legal_speed_policy.py`、`run_speed_calibration_point.sh`、`build_speed_calibration_design.py`、`speed_calibration_util085/policies/`。
- 后续注意：原策略仍用于质量代理的广覆盖 phase-error 训练；速度校准和所有 vLLM 实测仅使用 runtime-legal policy，保证 surrogate 的输入与真实运行一致。

## 2026-07-13 - Legal-policy runner ordering correction
- 开发目的：确保校准 exporter 实际接收合法投影而不是旧策略。
- 修改内容：修正 `RUN_GROUP` 在 `set -u` 下的初始化顺序；识别到先前已删除的 exporter 子进程仍写回了旧 checkpoint，待其退出后再次清除并用零非法 pair 的 p37/p39 policy 重启。
- 影响文件：`run_speed_calibration_point.sh`、`speed_calibration_util085/checkpoints/p37,p39`。
- 后续注意：每次 mixed checkpoint 导出后应检查 manifest 的 phase counts；若出现非合法 pair，即视为无效 checkpoint，不进入测量或拟合。

## 2026-07-13 - Quality proxy fit and predicted frontier
- 开发目的：在完整 72-policy WikiText suite 上冻结质量模型并生成待闭环的 Pareto 策略。
- 修改内容：合并完成 72/72 NLL shards；拟合 phase-local + global 质量代理，严格 holdout Spearman 为 0.835；生成速度校准设计与 predicted Pareto。修正 solver 展示项，移除拟合截距以确保 dense BF16 是精确 `ΔNLL=0` 锚点。
- 影响文件：`nll/prefill_decode.csv`、`reports/quality/`、`speed_calibration/design.csv`、`pareto/`、`solve_predicted_pareto.py`。
- 后续注意：Pareto CSV 仍是 screening 结果；曲线坐标必须使用后续 fresh E2E 和 actual-NLL closure，不可直接作为论文最终图。

## 2026-07-13 - Speed calibration design deduplication
- 开发目的：让速度校准只覆盖可执行且非重复的 phase policies。
- 修改内容：发现 p03 经 decode support 投影后与 p01 完全相同，替换为 p41；终止仍在运行的旧策略/重复 exporter，保留最新 legal p37/p39/p04，并启动 p41。
- 影响文件：`build_speed_calibration_design.py`、`speed_calibration/design.csv`、校准 job logs。
- 后续注意：p03 的旧测速仅是历史诊断，不能进入 `.85` legal-projection 速度校正拟合。

## 2026-07-13 - Mixed-export concurrency isolation
- 开发目的：避免两个异构 checkpoint 同时初始化 CUTLASS/Marlin 打包路径时长时间无有效进展。
- 修改内容：观察到 p37 已完成首层而 p39 尚未完成首层、两者均在 GPU 扩展初始化等待；停止未产出 p39，保留 p37 的已有转换进度，并将后续 mixed checkpoint 改为串行导出/测速。
- 影响文件：`logs/export_speed_p37.log`、`logs/export_speed_p39.log`、`speed_calibration_util085/checkpoints/`。
- 后续注意：此项仅改变导出调度，不改变策略、速度协议或质量标签；完成后必须验证 manifest 的合法 phase pair 和五次样本完整性。

## 2026-07-13 - Stale JIT lock recovery and SM120-only runner
- 开发目的：恢复被遗留 PyTorch extension lock 阻塞的 mixed export，并缩短后续 vLLM extension 初始化。
- 修改内容：确认 `torch_extensions/lock` 已失效而对应 `.so` 已生成；删除该临时锁后 p37 立刻由 1/32 推进至 32/32 并产出合法 checkpoint。测速 runner 显式固定 `TORCH_CUDA_ARCH_LIST=12.0`，避免对 RTX 5090 无关的 SM80/86/90/100 架构重复编译。
- 影响文件：`scripts/run_speed_calibration_point.sh`、`logs/export_speed_p37.log`、CUTLASS 临时 build artifacts。
- 后续注意：当前已启动的 p37 warmup 仍使用旧环境并会完成一次多架构 sparse extension 编译；新启动的测量点将使用 SM120-only 配置。

## 2026-07-13 - Reusable continuous-phase sampling path
- 开发目的：采用已有 `prepare_next_prefill()` phase restore 语义，避免每个重复 latency 样本重载异构 checkpoint。
- 修改内容：为 phase benchmark 增加可选的单 LLM 连续测量路径及逐样本 JSON 输出；新的 calibration policy 在一个实例内执行 1 次 warmup + 5 次连续 prefill-decode，并在每次之间恢复 prefill 权重。正在运行且已写入样本的 p37 保持原 protocol 完成，不中断。
- 影响文件：`artifacts/exports/vllm/ours/llama2-7b-chat/scripts/benchmark_phase_hetero.py`、`scripts/run_speed_calibration_point.sh`。
- 后续注意：所有用于同一速度拟合的数据仍记录逐样本 E2E，不将 LLM 初始化时间计入 latency；后续 closure 和任务评测也应优先使用此 continuous path。

## 2026-07-13 - Continuous-protocol calibration isolation
- 开发目的：消除每样本重载和单实例连续 restore 两种协议混用造成的速度校准偏差。
- 修改内容：p39 首次连续运行显示前两次 restore 仍在稳定期，因此连续 runner 改为 3 次不计入统计的 warmup + 5 次测量；新建 `speed_calibration_continuous085` 保存统一协议的样本，保留 `speed_calibration_util085` 作为历史 fresh-process 诊断。速度拟合器改读新的连续组，并允许重用旧组的 policy/checkpoint。
- 影响文件：`scripts/run_speed_calibration_point.sh`、`scripts/fit_speed_calibrator.py`、`speed_calibration_continuous085/`。
- 后续注意：必须在连续组中补齐所有 12 个设计点后才能拟合；不要将旧 `.85` 的重复重载样本混入该模型。

## 2026-07-13 - Phase restore steady-state gate
- 开发目的：确保连续 phase runner 报告的是 restore 后的稳态 latency，而非首次 materialization 的过渡成本。
- 修改内容：在 p39 上比较 3 与 6 次 warmup；6 次后五个样本为 3157.0--3162.7 ms，range 5.7 ms，而较短 warmup 的前两点明显偏慢。最终速度组更新为 `speed_calibration_continuous085_w6`，固定 6 warmups + 5 measured samples。
- 影响文件：`scripts/run_speed_calibration_point.sh`、`scripts/fit_speed_calibrator.py`、`speed_calibration_continuous085*/`。
- 后续注意：`continuous085` 与 `stabilization_p39` 是协议选择证据，不进入最终拟合；只有 `_w6` 的 12 个点进入 calibration CSV。

## 2026-07-13 - Final speed calibration and calibrated screening
- 开发目的：以连续 phase 的稳态 E2E 测量校准底层 roofline/kernel-sum 速度模型，并在策略 screening 中实际使用该校正。
- 修改内容：完成 12/12 个 runtime-legal design point 的 6-warmup + 5-sample suite；strict heldout MAE 从 raw dense-scaled 191.29 ms 降至 monotone calibration 的 59.18 ms。Pareto solver 现加载训练 split 的单调校准器，在每个候选上输出 `calibrated_predicted_e2e_ms` 与对应 speedup，同时保留 raw model 输出。
- 影响文件：`speed_calibration_continuous085_w6/calibration.csv`、`metrics.json`、`scripts/fit_speed_calibrator.py`、`scripts/solve_predicted_pareto.py`、`pareto/predicted_points.csv`。
- 后续注意：速度模型的数值预测已经过 holdout 验证；Pareto policy 仍需用 fresh vLLM E2E 和 actual WikiText NLL 进行 closure，才可用作最终图。

## 2026-07-13 - Measured Pareto closure
- 开发目的：将预测 screening 关闭为真实速度和真实 quality 的曲线，验证方法是否覆盖统一压缩对照。
- 修改内容：对 point_000/002/004/006/008/009 分别完成 100 WikiText blocks 的 phase-NLL 与 6-warmup + 5-sample continuous E2E；新增独立 closure runner、汇总 CSV/Markdown 与图。实测曲线从 1.000x/ΔNLL 0 到 1.692x/ΔNLL 2.882；在 dense-NVFP4 (1.058x, 2.882) 与 W4A16 (1.174x, 2.882) 附近，ours 有 1.096x/0.387、1.267x/0.543 的更优 trade-off。sparse-BF16 的 ΔNLL=55.3 以 off-scale 注释保留。
- 影响文件：`scripts/run_pareto_closure_speed.sh`、`scripts/make_closure_report.py`、`closure/nll/`、`closure/speed/`、`closure/summary.*`、`closure/report/pareto_measured_speed_nll.png`。
- 后续注意：该 closure 已证明 WikiText objective 的 Pareto 趋势；下一阶段应选择代表点跑生成任务数据集（并可将该图与 uniform baseline 一同用于论文）。

## 2026-07-13 - Persistent PMPD task-evaluation capacity fix
- 开发目的：以一个常驻 vLLM 实例完成一个生成任务 shard 的连续 prefill/decode，避免逐样本重载模型。
- 修改内容：为 PMPD evaluator 增加 `--max-model-len`，Llama 3.1 任务 runner 固定为 4096。烟测确认默认 131072 上下文会使 KV cache 申请 16.00 GiB、超过 0.85 显存配额下的 8.69 GiB，尚未进入生成；4096 覆盖本实验的输入与生成长度。
- 影响文件：`/home/agent/wja/project/my/cospaq/test/vllm/artifacts/dev/011_phase_switch_linear_test/pmpd_vllm_eval.py`、`artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/scripts/run_pareto_pmpd_shard.sh`。
- 后续注意：先以单样本确认 phase-switch task runner 成功，再将 point_004 的三个生成任务分片到 GPU 1--7。

## 2026-07-13 - Pareto representative downstream evaluation
- 开发目的：将实测速率 1.267x、WikiText ΔNLL 0.543 的 point_004 映射到真实生成任务，检验 proxy Pareto 趋势是否保留。
- 修改内容：新增 `run_full_pareto_pmpd.py`，按 CNN/DSum/IWSLT 的 128-sample shard 分发至 GPU 1--7；每个 shard 内 batch=16，使用单个 persistent PMPD vLLM 实例及 `prepare_next_prefill()` 完成连续阶段切换。单样本烟测已写出有效结果。
- 影响文件：`scripts/run_full_pareto_pmpd.py`、`scripts/run_pareto_pmpd_shard.sh`、`closure/tasks/point_004/`。
- 后续注意：全量完成后必须先逐 shard 校验样本数，再用既有 merger 汇总并运行 metrics-only；不要把 shard 层面的时延或模型加载时间作为正式 E2E 速度。

## 2026-07-13 - Two-point downstream Pareto closure
- 开发目的：以真实生成任务验证 WikiText 代理所选取的高质量与中间 Pareto 点，并给出可用于论文的任务级 trade-off 图。
- 修改内容：完成 point_002（1.096x, ΔNLL 0.387）和 point_004（1.267x, ΔNLL 0.543）在 CNN/DM-1000、DialogSum-1500、IWSLT-333 上的 persistent-PMPD 全量评测与 metrics-only 汇总；新增任务报告生成器、CSV、Markdown 和三张图。point_002 在 CNN/DSum/IWSLT 的主分数为 20.274/13.473/10.586，point_004 为 18.747/13.266/10.654；两点均覆盖 dense-NVFP4 与 sparse-BF16 的相关速度区域且保持显著更高的下游质量。
- 影响文件：`scripts/make_downstream_report.py`、`scripts/run_full_pareto_pmpd.py`、`scripts/run_pareto_pmpd_shard.sh`、`closure/tasks/point_{002,004}/results/quality/`、`closure/tasks/report/`。
- 后续注意：task 图的横轴来自同一 closure continuous protocol；uniform 的下游分数只读自冻结 baseline。point_009 的任务分数来自既有 max-speed run，作为 ours-max endpoint，不能与两项新评测混淆为同一生成吞吐 protocol。

## 2026-07-13 - Cross-scenario final handoff
- 开发目的：将已完成的 prefill-only 与 prefill-decode 证据整理为一个可复核、可写入论文的模型级结论。
- 修改内容：新增 Llama 3.1 最终摘要，链接两个场景的 measured frontier、任务级图、数值表和协议边界；明确下游 evaluator 的长生成 throughput 不是 formal E2E speedup。
- 影响文件：`artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/report/llama31_final_summary.md`。
- 后续注意：若新增模型或场景，应沿用该摘要的“正式场景速度 / 下游任务分数”分离表述，不混用 task generation throughput。
