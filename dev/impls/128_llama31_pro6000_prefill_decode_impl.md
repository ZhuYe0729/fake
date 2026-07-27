## 2026-07-22 - 完成 Llama3.1-8B-Instruct Pro 6000 prefill-decode 全流程

- 开发目的：在独立 066 目录复现并完成 Llama3.1-8B-Instruct 的 B8/S2048/O64 prefill-decode 实验，不修改历史实验目录。
- 修改内容：本地化全部脚本与 72 个策略；适配 Llama3 GQA 的 6144 宽融合 QKV 和 14336/28672 MLP shape；重建并验证两份 224-module canonical 稀疏状态；完成 local-error、72-policy NLL、质量拟合、8-shape kernel profile、Pareto 求解、26-point closure、11-policy Legacy PMPD 下游任务、汇总与七张图。
- 验证结果：`validation/all.json` 为 `ok=true`；NLL 72/72，closure 26/26，下游指标 33/33；holdout Spearman 0.9360；最快 closure 点 `point_020` 为 1.7404× dense BF16 speedup、实测 ΔNLL 0.35594。
- 影响文件：`artifacts/debug/066_llama31_pro6000_prefill_decode/`、`dev/plans/128_llama31_pro6000_prefill_decode_plan.md`、本文件。
- 后续注意：下游任务按用户选择冻结为 Legacy/raw-text PMPD 协议；Llama3 多数输出接近 256-token 上限，因此不得把本结果与 native chat-template 结果混用。canonical 为本机重新生成，历史 5090 hash 仅作信息记录。
