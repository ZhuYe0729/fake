## 2026-06-20 - FakeVLM Pareto search audit setup
- 开发目的：创建独立的 FakeVLM 小规模搜索 audit，用真实 prefill 速度和固定 20% FakeClue accuracy 检查 024 建模 Pareto frontier 的经验 gap。
- 修改内容：新增计划记录，准备在 `artifacts/debug/025_fakevlm_pareto_search_audit` 下实现 subset、policy search、combined validation、launcher 和 summary。
- 影响文件：`dev/plans/058_fakevlm_pareto_search_audit_plan.md`，`dev/impls/058_fakevlm_pareto_search_audit_impl.md`，`artifacts/debug/025_fakevlm_pareto_search_audit/`。
- 后续注意：不修改 `024_fakevlm_prefill_global_pareto`，仅作为 reference frontier 和 policy/runtime 代码来源。

## 2026-06-20 - Search audit scripts and CPU-side generation
- 开发目的：落地 025 search-audit 的可执行脚本，并完成 GPU 前的 CPU 侧产物生成。
- 修改内容：新增固定随机 20% subset 生成、三类 search policy 生成、combined speed+accuracy validator、6 GPU launcher、summary/gap 绘图脚本和 Slurm/local 启动脚本；生成 1000/5000 subset 和 60 个 batch16 search policies。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/`，`subset/`，`policies/`，`search/`，`summary/`。
- 后续注意：当前节点无法访问 NVIDIA driver，`nvidia-smi` 和 `torch.cuda.is_available()` 均失败；也没有 `sbatch/squeue`，因此真实 GPU validation 需要在 GPU/Slurm 节点上运行新增启动脚本。

## 2026-06-20 - Local GPU validation launch
- 开发目的：在当前可直接访问 GPU 的机器上启动 FakeVLM 小规模搜索审计。
- 修改内容：完成单配置 smoke test；启动并完成前 7 个真实验证结果；修正 launcher 状态日志，明确区分物理 GPU 与 worker 内部可见 GPU。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/launch_validation.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/logs/`，`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/`。
- 后续注意：worker 命令行会显示 `--gpu 0`，实际物理卡由 `CUDA_VISIBLE_DEVICES=<physical_gpu>` 隔离；批量验证可按已完成 json 自动跳过并继续。

## 2026-06-20 - Validation complete and summary
- 开发目的：确认 025 search audit 小规模验证完整跑完，并生成汇总结果。
- 修改内容：重跑 4 个因临时 CUDA OOM 失败的 random 配置；完成 60/60 policy validation；生成 search results、non-dominated frontier、gap-to-024 和速度-精度图。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/`，`artifacts/debug/025_fakevlm_pareto_search_audit/logs/launch_validation_retry_failed.log`。
- 后续注意：summary 显示 60 个搜索点中有 5 个非支配点，且在当前 20% subset validation 口径下有 6 个 024 reference 点被搜索点支配。

## 2026-06-20 - Revalidate 024 reference policies
- 开发目的：把 024 Pareto reference policies 放到 025 的固定 20% subset 和同一真实 speed/accuracy validator 下重测，避免旧 reference 与新搜索点口径不一致。
- 修改内容：新增 024 reference policy 导入脚本；将 8 个 batch16 reference policies 注册为 `reference_024` family；调整 summary 在存在实测 reference 时用实测 reference 画黑线和计算 gap；启动 8 GPU reference 重测。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/import_024_reference_policies.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/summarize_search.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/search/`，`artifacts/debug/025_fakevlm_pareto_search_audit/policies/reference_024/`。
- 后续注意：reference 重测后台 PID 为 382137；完成后结果数应新增 8 个 `reference_024*.json`，再运行 `summarize_search.py` 刷新 report 和图。

## 2026-06-20 - Reference revalidation complete
- 开发目的：确认 024 reference policies 已在 025 validator 口径下完整重测，并刷新最终对比。
- 修改内容：完成 8/8 个 `reference_024` policy validation；刷新 summary、measured reference、gap 表和速度-精度图。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/reference_024*.json`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/`，`artifacts/debug/025_fakevlm_pareto_search_audit/logs/launch_validation_reference_024.log`。
- 后续注意：刷新后的 summary 使用 `reference_source=measured_025`；60 个搜索点中仍有 5 个非支配点，6/8 个实测 024 reference 点被搜索点支配。

## 2026-06-20 - Add speedup visualization
- 开发目的：补充更适合展示的 speedup-vs-accuracy 视角，避免只用 latency 横轴。
- 修改内容：在 summary 脚本中新增 `search_speedup_vs_accuracy.png`，以实测 dense BF16 reference latency 为 1.0x 基准，横轴显示真实 E2E prefill speedup。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/summarize_search.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/search_speedup_vs_accuracy.png`。
- 后续注意：原始 `search_speed_vs_accuracy.png` 保留；新图横轴越右越快，更适合论文展示。

## 2026-06-21 - Revalidate corrected 024 frontier
- 开发目的：024 建模修正后重新评估更新的 batch16 Pareto policies，并保留旧图用于对照。
- 修改内容：将原 latency/speedup 图备份为 `_old.png`；归档旧 reference validation；重新导入当前 024 的点 `0,5,9,13,18,22,26,30`；完成 8/8 同口径实测并刷新 summary 和新图。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/summary/`，`artifacts/debug/025_fakevlm_pareto_search_audit/validation/reference_024_old/`，`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/reference_024*`，`artifacts/debug/025_fakevlm_pareto_search_audit/logs/launch_validation_reference_024_updated.log`。
- 后续注意：新 summary 仍有 6/8 个 reference 点被搜索点支配；最大同等/更高精度 latency improvement 为 30.49%，旧图可通过 `_old` 后缀查看。

## 2026-06-22 - Expand reference to refined sparse-BF16 v4
- 开发目的：将 025 大规模配置搜索对比中的 024 reference 更新为 `refined_sparse_bf16_v4` 扩展 frontier。
- 修改内容：扩展 reference 导入脚本以支持指定 024 report CSV；归档原 8 点 reference 和图；导入并完成 11/11 个 v4 policy 的同口径实测；刷新 measured reference、gap、latency 图和 speedup 图。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/import_024_reference_policies.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/search/reference_024_policies.csv`，`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/reference_024*`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/`。
- 后续注意：当前有效搜索结果为修正后已重测的 20 neighborhood + 10 suspicious；修正前 30 个 random 结果未混入新图。新 summary 为 30 个搜索点 + 11 个 v4 reference，4/11 个 reference 被搜索点支配，最大 latency improvement 为 25.78%。

## 2026-06-23 - Restore random search points
- 开发目的：在扩展 024 reference 对比图中重新纳入 random 搜索点，避免只展示 neighborhood/suspicious。
- 修改内容：确认归档 random policy 与当前 active random policy 的 module backend map 完全一致；恢复 30 个 random 实测 CSV/JSON；调整 summary 读取逻辑，用当前 `search_policies.csv` 覆盖策略元数据并保留实测 speed/accuracy；刷新 summary 和 speedup/latency 图。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/summarize_search.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/validation/policies/random_random_*`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/`。
- 后续注意：当前图为 60 个搜索点 + 11 个 `refined_sparse_bf16_v4` reference；random 实测结果来自归档，但 backend 分配与当前 random policies 一致，预测元数据由 current search table 覆盖。

## 2026-06-23 - Add presentation plot with uniform baselines
- 开发目的：为项目汇报/论文汇报补充包含 uniform baseline 的 speedup-accuracy 图，并将 reference 图例改为 `ours`。
- 修改内容：在 summary 脚本中新增 `search_speedup_vs_accuracy_with_uniform.png/pdf` 输出；从 024 `final_fakevlm_report_refined_sparse_bf16_v4.csv` 读取 uniform baseline 点，用紫色方块和方法名标注；保留原审计图不覆盖。
- 影响文件：`artifacts/debug/025_fakevlm_pareto_search_audit/scripts/summarize_search.py`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/search_speedup_vs_accuracy_with_uniform.png`，`artifacts/debug/025_fakevlm_pareto_search_audit/summary/search_speedup_vs_accuracy_with_uniform.pdf`。
- 后续注意：uniform baseline 点来自 024 refined sparse-BF16 v4 report；search/reference 点仍来自 025 固定 20% subset 同口径实测。
