# Qwen3 baseline-anchored decode repair

独立选点修正：保留 INT8 incumbent，加入部分 W4A16 decode，冻结24个 map、128窗口 validation 和64窗口 confirmation；沿用原 NLL 保护值与预算。当前状态见 results/worker.json；完成后见 REPORT.md。
