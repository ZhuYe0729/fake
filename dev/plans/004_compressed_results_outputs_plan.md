# 004 Compressed Results Outputs Plan

## 目标

- 将压缩模型的精度和速度结果从 dense 结果目录中拆出，避免 `maxvit_dense` / `dinov3_vit7b16_dense` 混放压缩方法结果。
- 修正 CSV 追加逻辑，避免已有表头不包含 checkpoint/compression 字段时产生无表头的尾部列。
- 支持多个 Slurm 作业同时追加同一个 CSV 时尽量保持结果文件结构一致。

## 方案

- 压缩评测 Slurm 脚本输出到 `artifacts/results/${MODEL}_compressed/accuracy.csv` 和 `speed.csv`。
- 保持 dense 评测脚本默认输出路径不变。
- 增强 `fake.utils.csv_io.append_csv_row`：
  - 新文件写入完整表头；
  - 已有文件表头缺字段时，按旧表头 + 新字段重写旧数据再追加；
  - 使用 lock 文件做进程间互斥，降低并发 Slurm 作业同时写入导致 CSV 损坏的风险。

## 验证

- 运行 Python 编译检查覆盖改动文件。
- 给出排除问题节点 `wqd10nah09g4` 的 MaxViT 压缩精度/速度重跑命令。
