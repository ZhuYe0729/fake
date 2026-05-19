# End-to-End Speedup PPT Plot Plan

## 目标
- 生成一页适合专业 PPT 汇报的端到端速度结果图，突出真实 CUTLASS 推理路径相对 dense baseline 的加速。
- 输出可直接放入 PPT 的 PNG/PDF，并保存配套 CSV 便于核对数据来源。

## 数据口径
- 使用 `artifacts/results/*/speed.csv` 中的端到端 forward throughput。
- Dense baseline 作为 `1.0x`。
- 展示真实可运行路径：Dense NVFP4、4:8 Sparse BF16、4:8 Sparse NVFP4。
- 不展示 Rescale/four-over-six 速度，因为当前 Rescale 结果来自 fake-quant 评估，没有真实 packed kernel 端到端速度。

## 实现步骤
1. 新增 `scripts/plot_end_to_end_speedup_summary.py`，读取 MaxViT tiny/small/base/large 和 DINOv3 的 speed CSV。
2. 按相同 batch size 计算每个模型内的 speedup，并输出 summary CSV。
3. 绘制 PPT 风格 grouped horizontal bar chart，标签包含 speedup 和吞吐。
4. 生成 PNG/PDF，并记录开发实现。

## 验证
- `python3 -m py_compile` 检查脚本语法。
- 在 `wja-cospaq` 环境下运行绘图脚本，确认 PNG/PDF/CSV 生成成功。
