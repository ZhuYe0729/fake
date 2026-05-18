## 2026-05-16 - CUTLASS runtime-packed checkpoint export and load
- 开发目的：将 DINOv3 CUTLASS dense/sparse NVFP4 从现场转换推理升级为 packed checkpoint 加载推理，明确真实压缩 checkpoint 口径。
- 修改内容：新增 runtime-packed checkpoint loader、导出脚本和 Slurm 入口；CUTLASS dense/sparse speed 与 accuracy 脚本支持 `--runtime-checkpoint`；CSV 记录 checkpoint format、runtime checkpoint 路径、source checkpoint 路径和 packed 文件大小。
- 影响文件：`dev/plans/013_cutlass_runtime_checkpoint_plan.md`、`fake/models/dinov3_cutlass_runtime.py`、`scripts/prepare_dinov3_cutlass_runtime_checkpoint.py`、`scripts/slurm/prepare_dinov3_cutlass_runtime_checkpoints.sh`、CUTLASS dense/sparse speed/accuracy 脚本与 Slurm 脚本。
- 后续注意：runtime loader 使用 meta skeleton + `assign=True` 加载 packed state dict，不调用 `from_linear()` 重新量化/pack；真实导出与推理验证需要在 RTX 5090 GPU 节点运行。

## 2026-05-16 - Runtime checkpoint Slurm and static checks
- 开发目的：补齐 packed checkpoint 端到端提交入口，降低 dense/sparse runtime checkpoint 混用风险。
- 修改内容：dense/sparse speed 与 accuracy Slurm 脚本支持 `RUNTIME_CHECKPOINT`；Python 脚本校验 runtime checkpoint backend；`scripts/README.md` 增加 packed checkpoint 导出和验证命令。
- 影响文件：CUTLASS dense/sparse speed/accuracy 脚本、对应 Slurm 脚本、`scripts/README.md`。
- 验证：通过 `python -m py_compile`、`bash -n`、`git diff --check`；登录节点可构建 DINOv3 meta skeleton，未分配真实 7B 权重。
