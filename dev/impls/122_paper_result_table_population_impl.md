# 122 Paper result table population implementation

## 2026-07-20 - Populate RTX 5090 paper table
- 开发目的：以已保留的 canonical 实测数据填写 `result.tex`，同时用 Max speed 与 Balanced 两个 Ours 行展示方法可达到的速度端与质量保持端。
- 选择：Llama2 prefill 为 `point_024` / `point_017`；Llama2 prefill-decode 为 `b8o64009` / `b8o64004`；Llama3 prefill 为 `point_014` / `bridge_dense_nvfp4_120`；Llama3 prefill-decode 为 `point_011` / `point_005`。前者均为该 track 内实测最快 Ours 点，后者优先选择质量接近强 uniform baseline 且仍具明显速度收益的点。
- 修改内容：填写 RTX 5090 两个模型的 prefill-only 质量/TTFT 与 prefill-decode 三任务质量、TTFT、TPOT、E2E speedup；uniform decode speed 来自同场景 canonical `uniform_baselines.csv`，Ours decode speedup 以该 BF16 baseline 重新计算。RTX PRO 6000 保留 `--`，因为无可用实测结果。
- 验证结果：检查所有数据行均为 12 个 `&`（13 列），数值统一四舍五入至两位；Llama2 BF16 ARC-E 更正为源数据的 68.39。
- 后续注意：Llama2 canonical prefill-decode track 未保留 uniform 生成任务分数，因此其 uniform 的 CNN/DM、DialogSum、IWSLT 单元格如实维持为 `--`；不得用不同配置的旧实验补填。
