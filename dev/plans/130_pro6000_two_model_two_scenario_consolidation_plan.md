# Pro 6000 两模型两场景结果汇总计划

## 目标

在不修改 060、064--067 的前提下，将 RTX PRO 6000 上 Llama-2-7B-Chat 与
Llama-3.1-8B-Instruct 的 prefill-only / prefill-decode 正式结果汇总到新的 068
独立目录，并基于 060 的 `result_v2.tex` 生成填入 Pro 6000 结果的
`result_v3.tex`。

## 假设与口径

- 064--067 的 `validation/all.json` 均为 `ok=true`，其
  `results/complete_results.csv` 是本次唯一数值来源，不重新跑下游任务。
- Max-speed 只在 ours 且表格所需下游指标齐全的点中选择实测速度最高者。
- Prefill-only Balanced：四项准确率相对 BF16 的平均保持率不低于 95%，且任一项
  不低于 90%；在合格点中选择实测速度最高者。
- Prefill-decode Balanced：三项生成指标相对 BF16 的平均保持率不低于 87.5%，且
  任一项不低于 75%；在合格点中选择实测 E2E 速度最高者。
- 065/066 仅保存 O=64 的 E2E 计时，没有独立 TTFT/TPOT 实测。因此
  `result_v3.tex` 不用预测值或跨 checkpoint 推算值冒充实测值；Pro 6000 的 decode
  TTFT/TPOT 保持 `--`，并在 caption 和汇总说明中明确这一限制。

## 实施步骤与验证

1. 核验四个源 bundle 与 SHA-256 → 验证四个 `validation/all.json` 均通过。
2. 创建 068 的自包含汇总脚本和源 CSV 快照 → 验证快照 SHA 与源文件一致。
3. 按固定规则生成选择记录、合并表和 `result_v3.tex` → 验证选中点确为规则下最快点。
4. 生成 README 与机器可读验证结果 → 验证 Pro 6000 的质量、prefill TTFT、decode
   E2E 均无空值，decode TTFT/TPOT 仅按已声明口径保留缺测标记。

