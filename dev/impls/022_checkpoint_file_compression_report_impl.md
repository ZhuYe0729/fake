## 2026-05-18 - Real Checkpoint File Compression Report
- 开发目的：区分 metadata 估算压缩率和真实 checkpoint 文件大小压缩率，避免汇报中混淆。
- 修改内容：新增 `scripts/report_checkpoint_file_compression.py`，扫描 dense source 和 `artifacts/checkpoints/**/model.pt`，输出真实文件大小比值 CSV/Markdown，并标注 fake dense state_dict 与 real packed checkpoint。
- 影响文件：`scripts/report_checkpoint_file_compression.py`、`artifacts/results/checkpoint_file_compression_ratios.csv`、`artifacts/results/checkpoint_file_compression_ratios.md`。
- 后续注意：当前 Rescale checkpoint 仍是 fake dense state_dict，不代表真实存储压缩；真实 packed Rescale checkpoint 需要后续 packer/runtime 支持。
