## 2026-06-15 - DINOv3 layerwise max-speed scaffolding
- 开发目的：实现基于速度模型的 DINOv3 分层最优 CUTLASS 后端选择，并能在超算 GPU 节点实测整模型 forward 速度。
- 修改内容：在 `artifacts/debug/019_dinov3_layerwise_max_speed/code/` 下新增 DINO layerwise policy 替换模块、策略生成/benchmark/summary runner、Slurm 脚本和 053 plan 文件。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/dinov3_layerwise_policy.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh`、`dev/plans/053_dinov3_layerwise_max_speed_plan.md`。
- 后续注意：本机缺少 DINO 权重时只能做静态验证；完整速度需要在 RTX 5090 超算节点运行 Slurm 脚本。

## 2026-06-15 - 超算路径导入修复
- 开发目的：修复超算运行时报 `ModuleNotFoundError: No module named 'modeling'`，并避免 Slurm 脚本硬编码仓库路径。
- 修改内容：runner 从脚本位置向上发现仓库根目录，并自动搜索 `fake/**/modeling/kernel_predictor.py` 后加入 import path；Slurm 脚本改为从自身位置反推 `REPO_ROOT` 后进入仓库。
- 影响文件：`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.py`、`artifacts/debug/019_dinov3_layerwise_max_speed/code/run_dinov3_layerwise_max_speed.sh`。
- 后续注意：如果超算仓库确实缺少 `fake/kernels/cutlass/cutlass_wrapper/modeling/kernel_predictor.py`，新报错会打印搜索过的路径。
