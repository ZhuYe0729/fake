# Pro 6000 两模型两场景结果汇总开发记录

## 2026-07-23 - 完成 064--067 汇总与 result_v3

- 开发目的：汇总 RTX PRO 6000 上两模型、prefill-only / prefill-decode 四组正式结果，补全论文总表中的 Pro 6000 主结果。
- 修改内容：新增 068 独立目录；冻结四份源 CSV 与 validation 快照；实现基于 BF16 质量保持率的 Max-speed/Balanced 可复现选点；生成 28 行选择表、来源/SHA 记录、验证文件和 `result_v3.tex`。
- 影响文件：`artifacts/debug/068_two_model_two_scenario_pro6000_result_consolidation/`；未修改 060、064--067。
- 验证：四个源 validation 均通过，CSV/validation 快照哈希一致；独立审计确认八个 ours 点符合规则、5090 表格区域逐字不变、Pro 6000 共 14 行且每行列数正确。
- 后续注意：065/066 未保留 decode TTFT/TPOT 分项实测，因此 v3 对非 BF16 行明确保留 `--`，仅填入实测 E2E；若论文必须报告分项，需要另建实验测量，不应使用 predictor 或跨 checkpoint 推算冒充实测。066 下游结果为冻结的 Legacy/raw-text PMPD prompt 口径。

## 2026-07-23 - 补齐 RTX 5090 同口径 decode TTFT/TPOT/E2E

- 开发目的：在 RTX PRO 6000 上按 RTX 5090 的原始协议补齐两个模型、14 个论文表策略的 decode 分项速度，消除 `result_v3.tex` 中的 TTFT/TPOT 缺测项。
- 修改内容：在 068 内新增 checkpoint 导出/校验、逐样本独立 vLLM 进程测量、汇总和全量审计脚本；O=1 与 O=64 均采用 1 次 warmup + 5 次 measured，TPOT 按两个中位数之差除以 63；回填 14 行实测 TTFT/TPOT/E2E speedup。
- 影响文件：仅更新 `artifacts/debug/068_two_model_two_scenario_pro6000_result_consolidation/` 和本开发记录；060、064--067 未修改。
- 验证：14 个策略的 checkpoint audit 均通过；当前 168 份原始样本对应 168 个独立进程且 GPU UUID 一致；TTFT/E2E 最大 CV 为 0.64%/0.78%；临时 checkpoint 文件为 0；`validation/decode_components.json` 与 `validation/all.json` 均为 `ok=true`。
- 稳定性处理：Llama-3.1 balanced 首轮 TTFT CV 为 4.67%，按同一协议整组重测后为 0.64%；首轮 6 个原始样本与 summary 保留在 `superseded_unstable_ttft/`，未挑选单次最优值。
- 结果变化：同口径 E2E 下，Llama-2 max/balanced 为 1.45×/1.32×，Llama-3.1 max/balanced 为 1.61×/1.34×；与 065/066 单模型常驻进程测得的旧 E2E 数值不同，论文表统一使用本次 RTX 5090 同口径结果。
