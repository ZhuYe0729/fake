## 2026-05-18 - Rescale Accuracy Compression Figure
- 开发目的：生成用于专业 PPT 汇报的压缩率与精度汇总图，突出 rescale 方法相对原始 NVFP4 sparse 的提升。
- 修改内容：新增 `scripts/plot_rescale_accuracy_compression_summary.py`，按原始方法 worst seed、rescale 方法 activation off best seed 的口径汇总 MaxViT tiny/small/base/large 和 DINOv3，输出 CSV、PNG、PDF。
- 影响文件：`scripts/plot_rescale_accuracy_compression_summary.py`、`artifacts/results/rescale_accuracy_compression_summary.*`。
- 验证：`python3 -m py_compile` 通过；在 `wja-cospaq` 环境下成功生成 PNG/PDF/CSV。
- 后续注意：图中不使用 four over six 命名，统一标注为 `Rescale`。

## 2026-05-18 - Switch Plot CR to Real File Ratio
- 开发目的：按真实 checkpoint 文件大小压缩率重新绘制当前汇总图，避免把 metadata 估算 CR 误认为真实文件压缩。
- 修改内容：图中 `CR` 改为 `File CR`，计算方式为 dense source bytes / selected checkpoint `model.pt` bytes；图注同步改为 real checkpoint file-size compression ratio。
- 影响文件：`scripts/plot_rescale_accuracy_compression_summary.py`、`artifacts/results/rescale_accuracy_compression_summary.*`。
- 后续注意：当前 Original NVFP4/Rescale 选择结果对应 fake dense state_dict 或 no-checkpoint eval，因此真实文件大小 CR 约为 1.00x；真实 packed storage CR 需使用 `cutlass_*` checkpoint 或后续实现 Rescale packer。

## 2026-05-18 - Use Original NVFP4 Effective CR for Rescale
- 开发目的：根据 rescale 可吸收到 NVFP4 scale 中、不引入额外存储的假设，修正汇报图中的压缩率口径。
- 修改内容：Rescale 柱子的 CR 标签与 CSV 中 `compression_ratio` 复用同模型、同 sparse family 的 Original NVFP4 有效存储 CR；图注说明 CR 使用对应 Original NVFP4 effective storage ratio。
- 影响文件：`scripts/plot_rescale_accuracy_compression_summary.py`、`artifacts/results/rescale_accuracy_compression_summary.*`。
- 后续注意：真实 checkpoint 文件大小报告仍保留在 `artifacts/results/checkpoint_file_compression_ratios.*`，但当前 PPT 图不使用 fake checkpoint 文件大小作为 rescale CR。

## 2026-05-18 - Use Real Non-Rescale Packed Checkpoint CR
- 开发目的：按真实压缩后 checkpoint 文件大小重新给汇报图标注 CR，同时保持 Rescale 不额外增加存储的口径。
- 修改内容：绘图脚本读取 `artifacts/results/checkpoint_file_compression_ratios.csv` 的 `file_size_ratio`；Rescale 复用对应非 Rescale packed checkpoint 的真实 CR，并在输出 CSV 中增加 `compression_ratio_source` 记录来源 checkpoint。
- 影响文件：`scripts/plot_rescale_accuracy_compression_summary.py`、`artifacts/results/rescale_accuracy_compression_summary.*`。
- 后续注意：当前 4:8 structured 使用 `cutlass_sparse_nvfp4_storage` 的真实文件 CR；unstructured 面板使用现有非 Rescale `cutlass_nvfp4_runtime` 的真实文件 CR，若后续实现 unstructured sparse packed checkpoint，需要替换该来源。
