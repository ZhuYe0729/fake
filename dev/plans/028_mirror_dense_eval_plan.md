# MIRROR Dense 鉴伪评测计划

- 开发目的：跑通 MIRROR 原始 dense/full-precision 鉴伪模型在 Chameleon 与 GenImage validation 上的完整评测流程。
- 修改内容：新增项目内 MIRROR 评测入口，复用 `third_party/MIRROR` 模型定义和本地权重；适配 Chameleon 解压目录与 GenImage validation zip；新增 Slurm 提交脚本和命令文档。
- 影响文件：`scripts/eval_mirror_dense_accuracy.py`、`scripts/slurm/eval_mirror_dense_accuracy.sh`、`scripts/slurm/all_model_test_commands.md`、`dev/impls/028_mirror_dense_eval_impl.md`。
- 后续注意：本计划只评测原始 MIRROR，不接入任何压缩、量化或剪枝流程；默认在 GPU 计算节点运行，登录节点只做静态与数据发现检查。
