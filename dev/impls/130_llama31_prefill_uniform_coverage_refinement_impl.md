# 130 Llama3.1 prefill-only uniform-coverage refinement implementation

## 2026-07-22 - Experiment bootstrap and fine-budget screening
- 开发目的：在独立目录中补充 Llama3.1 prefill-only 策略点，尝试以真实速度和 ARC-Challenge 包络 uniform Dense NVFP4。
- 修改内容：新增 debug 068；冻结 058/060/061 来源及 SHA256；准备 bridge-104/120、point-008 seed；复用 061 warmed speed protocol 编写独立速度、NLL 与 ARC-Challenge runner；在原 058 DP 求解器上将 p007--p009 之间的质量预算加密为 22 个唯一候选。
- 验证结果：脚本通过 `py_compile`；source policy 拷贝哈希与来源一致；细粒度求解结果中 `fine_q196`--`fine_q220` 的校准后预测 E2E 为 615.34--598.29 ms，覆盖 Dense-NVFP4 的候选阈值区间，等待真实 GPU closure。
- 后续注意：GPU 首次启动请求因沙箱外审批服务连接中断被拒绝，尚未实际启动任何 068 GPU 任务；需要用户在获知该情况后重新明确授权。

## 2026-07-23 - Measured closure and downstream validation
- 开发目的：在冻结的 061 warmed runtime 口径下补齐真实速度、real-vLLM NLL 与下游任务，验证能否包络 uniform Dense NVFP4。
- 修改内容：顺序复测 bridge-120、point-008 和 fine-q196/q202/q208/q214 的五次 E2E 延迟；测量四个 fine policy 的 real-vLLM NLL；全量评测 ARC-Challenge，并为 q202/q208 补齐 WikiText、WinoGrande、ARC-Easy、MMLU；新增全指标覆盖审计和清理后的论文图。
- 验证结果：`fine_q208` 实测 `1.79554x / 0.52048 ARC-Challenge acc_norm`，严格优于 uniform Dense NVFP4 的 `1.75736x / 0.51109`；`fine_q202` 也以 `1.77412x / 0.51280` 严格优于该基线。程序化 ARC 审计显示全部五个 uniform 点均被 ours 包络。q208 同时覆盖 Dense NVFP4 的 WinoGrande 和 ARC-Easy，但 WikiText PPL 与 MMLU 未覆盖；失败结果保留在 `all_results.csv` 与 `all_metric_coverage.csv`。
- 影响文件：`artifacts/debug/068_llama31_prefill_uniform_coverage_refinement/` 下的 policy、速度/NLL/任务原始 JSON、汇总 CSV、`summary.md` 和 `pareto_arc_challenge.png`。
- 后续注意：ARC-Challenge 是本次预先选定的论文绘图指标；不应宣称 q208 在所有 prefill-only 指标上均支配 Dense NVFP4。058 的 bridge ARC-Easy 仅有 `acc_norm`，汇总时已置空，避免与当前论文协议的 `acc` 混用。

## 2026-07-23 - Unified two-scenario Llama3.1 paper figures
- 开发目的：让论文中的 prefill-only 与 prefill-decode trade-off 图统一使用 Llama3.1-8B-Instruct，并采用相同视觉与严格帕累托判定。
- 修改内容：新增 `build_two_scenario_paper_figures.py`；prefill-only 使用 068 ARC-Challenge，prefill-decode 使用 060/057 CNN/DM ROUGE-L；生成两张独立 PNG/PDF 和一张双面板 PNG/PDF；增加逐 uniform dominance 审计。
- 验证结果：两个场景各五个 uniform baseline 均由至少一个 ours 实测点支配。prefill-decode 的 `point_007` 仍保留在源表，但因源报告明确标记速度异常而不进入论文图和边界。
- 修正事项：057 旧绘图函数按速度升序保留“历史最高质量”，不等价于最大化速度与质量的二维非支配集合，因而旧边界线不完整；新脚本改为逐点 pairwise dominance，未修改任何测量值。
- 影响文件：`artifacts/debug/068_llama31_prefill_uniform_coverage_refinement/llama31_8b_instruct/report/paper/`、068 汇总文档及 060 Llama3.1 prefill-decode 汇总中的链接说明。

## 2026-07-23 - Frontier-only plotting and IWSLT main panel
- 开发目的：减少主图视觉噪声，并选择更清楚展示 prefill-decode trade-off 的主指标。
- 修改内容：主图只绘制严格非支配 ours 点，不再显示被支配的灰色点；prefill-decode 主面板由 CNN/DM ROUGE-L 切换为 IWSLT SacreBLEU；CNN/DM 图继续作为独立附录备选。
- 验证结果：IWSLT 下五个 uniform baseline 仍全部被 ours 严格包络；双面板图的覆盖审计仍为 10/10，且 IWSLT 边界提供从 `11.64` 到 `1.29` SacreBLEU、`1.03x` 到 `1.42x` 的连续实测 trade-off。
- 后续注意：选择 IWSLT 是论文主图的展示选择，不改变 CNN/DM、DialogSum 或其他已报告任务结果；正文/附录应保留完整任务表以避免产生只报告单一最优指标的印象。

## 2026-07-23 - Prefill-only midrange frontier densification
- 开发目的：填补 prefill-only 约 1.2--1.7x 区间的论文曲线空档，同时坚持只绘制真实非支配点。
- 修改内容：先闭环 point-004/006、fine-q124/160/190 和 dense-NVFP4 bridge-088/096；发现固定 bridge 与较高预算策略大多被支配后，将原 DP 求解器参数化，并在 0.04--0.08 质量预算之间生成 low-q 细粒度候选；顺序测五次 warmed E2E、并行测全量 ARC-Challenge。
- 统一口径修正：新测 bridge-088 与 058 旧 ARC 结果明显不一致，因此补测 bridge-120，并从汇总脚本中移除 `068 speed + 058 quality` 混合。bridge-120 当前真实 ARC `acc_norm=0.52986`，而非旧表的 `0.53840`；所有论文曲线现均使用 068 当前真实 runtime 结果。
- 验证结果：新增真实边界点为 point-004 (`1.2523x/0.5461`)、low-q046 (`1.2862x/0.5410`)、low-q062 (`1.4373x/0.5341`)、point-006 (`1.5042x/0.5324`) 和重测 bridge-120 (`1.7338x/0.5299`)。其余尝试完整保留在 `all_results.csv`，但因被支配不进入主图。
- 影响文件：068 的策略、速度/ARC 原始 JSON、`predicted_low_candidates.csv`、`midrange_bridge_candidates.csv`、汇总脚本、覆盖审计以及全部 paper PNG/PDF。
- 后续注意：不要为了视觉密度恢复被支配点；当前较长的 1.50--1.73x 线段反映了该离散 action space 下的真实质量跳变。
