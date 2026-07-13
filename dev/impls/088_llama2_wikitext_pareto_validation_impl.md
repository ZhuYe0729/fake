## 2026-07-11 - Selected predicted-frontier NLL validation launch
- 开发目的：在运行昂贵的 vLLM 导出/E2E 前，先检查预测 frontier 的真实 WikiText NLL 排序。
- 修改内容：新增端点/膝点/中间点选择器和独立 policy JSON 的 NLL 验证入口；每场景选择 5 点。因最快端点同时是膝点，补充中位点以保持 5 个验证点。
- 验证：已启动 prefill-only 5 点的 100-block WikiText NLL（GPU 0–4，一卡一 7B worker）；之后自动运行 prefill-decode。
- 后续注意：只有验证排序合理，才导出 selected policy checkpoint 并启动 vLLM E2E；PMPD 仍不参与本轮拟合。

## 2026-07-11 - 统一 decode 速度口径与基线补测准备
- 开发目的：避免将旧 `gpu_memory_utilization=0.9` baseline 数字和 phase-hetero decode 的 `0.8` 数字混入同一论文曲线。
- 修改内容：将候选 decode 速度结果统一保存到 `speed_mem08/`；新增 `run_uniform_reference_vllm_speed.sh`，在候选测量释放 GPU 后，对 sparse-bf16、w4a16 与 sparse-nvfp4 的可运行阶段对（prefill sparse-nvfp4 / decode dense-nvfp4）使用完全相同的 fresh-process、batch=16、2048+80、0.8 显存协议测量。dense-bf16/dense-nvfp4 分别由 selected point 0/11 覆盖。
- 影响文件：`artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/scripts/run_uniform_reference_vllm_speed.sh`。
- 后续注意：该脚本必须在当前候选单卡速度任务结束后运行，不能并发占用 GPU 7。

## 2026-07-11 - Decode 实测对比汇总器
- 开发目的：用真实质量和同 runner 的速度判定最终 Pareto，而不是沿用预测值或混用旧 baseline 配置。
- 修改内容：新增 `build_decode_comparison.py`；它读取 5 个 selected point 的真实 WikiText NLL 与 `speed_mem08` 的 E2E/TTFT/TPOT 中位数，并合并三种待补测统一参考，输出全局支配标记。
- 验证：Python 编译检查和 shell 语法检查通过；完整运行依赖尚在执行的速度结果。

## 2026-07-11 - Prefill-only 全局实测前沿
- 开发目的：以真实 WikiText NLL 和 baseline-aligned 实测速率判断 prefill-only 的全局前沿。
- 修改内容：新增并执行 `build_prefill_comparison.py`，输出 `validation/prefill_only/measured_comparison.csv`，合并 5 个 selected point 与 5 个统一参考并计算支配关系。
- 验证：真实保留的 ours 点为 4、8、16（NLL 增量 0.0011/0.0047/0.7359，对应 1062.2/890.0/528.4 ms）；point 0 与 12 被 dense-bf16 和 dense-nvfp4 参考支配，已明确标记，不会被用于宣称 Pareto。
- 后续注意：dense-bf16、dense-nvfp4、sparse-nvfp4 仍是全局前沿上的合法退化策略；论文图应将它们显示为单一方法参考，而不是错误宣称所有参考都被异构点严格支配。

## 2026-07-12 - Decode 统一参考补测与论文汇总
- 开发目的：以完全同一 phase-runtime 协议检验 decode 曲线能否覆盖其余可运行的统一方法，并输出可复核的论文图表。
- 修改内容：完成 sparse-bf16、合法 sparse-nvfp4 阶段对（prefill sparse-nvfp4 / decode dense-nvfp4）和 w4a16 的 3×22 次 fresh-process 测量；运行 `build_decode_comparison.py`；新增并运行 `build_paper_report.py`，生成 `report/measured_pareto.png` 与 `report/README.md`。
- 验证：decode 的 five selected ours points 均为全局实测非支配点。sparse-bf16（ΔNLL 29.1159, 5240.2 ms）、合法 sparse-nvfp4 阶段对（3.0959, 4712.3 ms）和 w4a16（2.1151, 3725.7 ms）均被 ours 支配；其中 ours point 11 在与 w4a16 相同 ΔNLL 下 E2E 为 3499.9 ms。
- 后续注意：prefill-only 与 decode 的 baseline 口径差异已在报告中明确；后续若扩展 frontier，只能使用相同 scenario/runner 再加入数据点。

## 2026-07-12 - 论文图的 speedup-质量重构
- 开发目的：以论文中更直观的“横轴越右越快、纵轴越上质量越好”形式，突出混合策略曲线与单一方法点的覆盖关系。
- 修改内容：新增 `build_paper_speedup_plots.py`，输出 prefill-only 与 prefill-decode 两张独立的大尺寸图；横轴使用相对 dense-BF16 的实测 E2E speedup，纵轴使用 `-ΔNLL`。ours 为深色连线圆点，统一方法为红色方块。decode 的 sparse-BF16 因 ΔNLL=29.1 采用明确的 off-scale 下三角标记，避免压扁有效 frontier。
- 验证：人工检查两张 PNG 可读性；decode 图清楚显示 ours point 11 以同等质量且更高 speedup 覆盖 W4A16，以及混合曲线覆盖合法 sparse-NVFP4 阶段对。

## 2026-07-12 - 显式标注 max-speed 端点
- 开发目的：避免深色 Pareto 曲线最右端的 max-speed 解被误读为未包含在图中。
- 修改内容：在两张 speedup 图中将求解器的最大质量约束端点以金色星形及 `Ours max-speed` 标注。prefill-only 对应 point 16（2.04×，ΔNLL=0.736）；prefill-decode 对应 point 11（1.40×，ΔNLL=2.115；prefill dense-NVFP4 / decode W4A16）。
- 验证：重新渲染并人工检查两张 PNG；星形端点清晰且保留原始支配关系。进一步逐 module 比对确认 point 16/11 分别与此前导出的 `max_speed` prefill-only/prefill-decode policy 完全相同。
