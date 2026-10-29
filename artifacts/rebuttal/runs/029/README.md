# Qwen3条件增量质量校准

清理153之后的新DASO诊断：固定SQ-INT8 prefill，36个预定校准map，16个decode组和8个pair交互。仅用policy holdout确定主模型，再生成至多20个提案/anchor，128新validation与128新confirmation。状态见results/worker.json，完成后见REPORT.md。
