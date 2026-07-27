# Llama2 当前 GPU 质量/速度代理与 Pareto 流程计划

## 目标

将根目录 `profile.sh` 改为当前 GPU 专用、可追溯的 Llama2-7B-Chat 校准入口：重新收集 033 的 phase-local quality 数据、重建 kernel predictor、以新 proxy 求解 034 候选，并明确区分后续 035/037 的 E2E 校准闭环。

## 实施步骤

1. 修正 profile 脚本的 CUDA 可见设备、路径与隔离输出目录；所有 kernel profile 都在同一物理 GPU 完成。
2. 修正 034 solver 的 wrapper 导入路径，并让它接收指定 predictor root 和 033 phase-local errors，避免读取历史机器的 007 数据。
3. 在 profile 脚本中串联 033 的 inputs、local errors、NLL shard/merge、quality fit、kernel profile、034 candidate solve；将 E2E 校准保留为候选产生后的显式运行阶段。
4. 进行 shell/Python 语法检查与 CLI smoke 验证，不运行耗时的全量 NLL 或 vLLM E2E 任务。

## 成功标准

- 不再混用物理 GPU 或历史 predictor/model artifacts。
- solver 读取本次 profile 的模型和本次 033 的 phase-local error。
- 每个阶段输出路径和完成条件在脚本中清晰可检查。
