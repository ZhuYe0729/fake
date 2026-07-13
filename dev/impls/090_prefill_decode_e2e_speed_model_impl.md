## 2026-07-12 - Debug model plan initialized
- 开发目的：解决 prefill-decode 中 kernel 线性速度模型对端到端 vLLM 速度过度乐观的问题。
- 修改内容：建立 plan 090；后续测量、特征和拟合均写入独立的 `artifacts/debug/035_llama2_prefill_decode_e2e_speed_model/`。
- 后续注意：不使用 `.8` provisional runner 数据，也不覆盖 034 的结果图或策略。

## 2026-07-12 - Formal E2E calibration launched
- 开发目的：为 decode 专用端到端速度校正收集覆盖压缩比例的正式测量，而不是复用不兼容的 `.8` 数据。
- 修改内容：新增每 GPU 独立的 checkpoint 导出/正式 runner 脚本、测量汇总脚本和单调 E2E 校准器；GPU 0–6 并行补测 point 1、2、4、5、7、9、10，每点 output 1/80 各 3 次 fresh-process。
- 初步发现：point 7、9 与此前 point 8 一样在 `.9` 正式配置下，NVFP4 prefill workspace OOM；point 11 却可行，说明必须为策略求解增加显存可行性约束，且该约束不是简单的压缩比例单调函数。
- 后续注意：point 1、2、4、5、10 仍在执行，完成后再进行 leave-one-out 校准验证；尚未用新模型重求 Pareto。

## 2026-07-12 - E2E calibration completed
- 开发目的：验证端到端单调校正能否修正 kernel 延迟线性相加的系统性偏差。
- 测量结果：新增 point 1、2、4、5、10 各完成 output 1/80 的 3 次 fresh-process；连同已有正式 10 次锚点 0、3、6、11，共得到 9 个可行策略。point 7、8、9 明确 OOM。
- 建模结果：以 point 0 单点缩放的 raw linear 模型在这 9 点平均绝对误差为 326.0 ms；单调 E2E 校正的 leave-one-out MAE 为 90.8 ms。其保留速度排序先验，同时避免把小比例异构压缩错误地预测为明显加速。
- 后续注意：该校正仍是 debug 模型，尚未进入 Pareto 求解。下一步应实现并验证显存可行性约束，再用 corrected E2E + feasibility 重新选择策略点。

## 2026-07-12 - KV headroom OOM workaround validated
- 开发目的：区分策略本身不可行与 vLLM KV-cache 预留导致的 prefill 峰值 OOM。
- 验证方法：保持正式 phase runner、batch 16、2048+80 和全量 32768-token prefill 不变，仅把 `gpu_memory_utilization` 从 `.9` 调至 `.85`，对原本 OOM 的 point 7、8、9 各做一次 output-80 探针。
- 验证结果：三点均成功，单样本 E2E 分别为 point 7=4431.955 ms、point 8=4266.485 ms、point 9=3493.083 ms。因此 OOM 由 KV-cache 预留挤占 NVFP4 activation packing/GEMM 峰值空间所致，不应作为策略不可行标签。
- 后续注意：`.85` 是新的部署协议，必须在该配置下重新测 dense 基线、所有候选点和 uniform references，不能与 `.9` 测速表混用。

## 2026-07-12 - Full `.85` formal curve sweep
- 开发目的：在已验证可行的统一 KV-cache 配置下补齐全部 12 个预测 Pareto 候选点。
- 修改内容：校准 runner 增加独立输出组、可配置 GPU memory utilization，并复用既有 checkpoint；启动 point 0–11 的 output 1/80 各 10 次 fresh-process 测量。
- 当前结果：point 0–8、10、11 完成且无策略 OOM。point 9 首次在 GPU 复用时因前序 vLLM 进程尚未释放显存而启动拒绝，不是策略 OOM；已在独立 GPU 7 重跑。
- 后续注意：point 9 完成后汇总 `.85` 曲线；uniform baseline 仍需在同一配置补测。

## 2026-07-12 - `.85` curve visualization
- 开发目的：先以统一 `.85` 实测速度检查 OOM 窗口消除后的完整策略曲线。
- 修改内容：新增 `build_util085_plot.py`，输出全部 12 点的正式速度汇总及 Pareto 图。因只有部分点有真实 WikiText NLL，图的纵轴明确使用 solver 的 predicted quality cost；真实质量图不与此混用。
- 结果：point 0–8、10、11 的 10 次中位数已进入曲线；point 9 同时存在 3245–7189 ms 的并发干扰，按用户要求先显示为不稳定点、且不参与 frontier。`.85` 下 max-speed point 11 为 3004.984 ms，相对同协议 point 0 为 1.714x。
- 后续注意：补测同协议 uniform baselines 与真实质量点前，图只能用于速度/求解诊断，不能作为最终论文精度曲线。

## 2026-07-12 - Full actual WikiText NLL launch
- 开发目的：将 `.85` 实测速度图的预测纵轴替换为全部候选点的真实质量测量。
- 修改内容：新增单 GPU NLL runner；使用既有 100-block WikiText teacher-forced pooled-NLL 协议，对 point 0–11 的 prefill 与 decode 分别测量后计算 `target_delta_nll`。已在 8 GPU 并行启动。
- 后续注意：完成后以真实 ΔNLL 重绘 `.85` 图；point 9 的速度仍保留不稳定标记，不能因质量测量完成而加入最终速度前沿。

## 2026-07-12 - Actual-NLL Pareto figure completed
- 开发目的：完成 `.85` 统一速度协议下、全部策略点使用真实 WikiText 质量的图示。
- 测量结果：12 个 point 的 100-block pooled `target_delta_nll` 全部完成；第二批 GPU 复用导致的评测 OOM 通过分配空闲 GPU 单独重跑解决，非策略 OOM。
- 修改内容：更新 `.85` 作图脚本为读取 `actual_nll/point_*.csv`，输出 `pareto_speedup_vs_wikitext_prefill_decode_util085_actual_nll.png` 和对应汇总 CSV。
- 后续注意：point 9 的正式速度仍受并发影响（3245–7189 ms），图中明确保留为不稳定点且不参与 frontier；uniform baselines 尚待同协议补测。
