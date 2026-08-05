# 131 Llama3.1 single-column paper figures plan

## 目标

将 debug 068 `report/paper/` 中的全部论文图按单栏排版重新绘制，在不修改任何实验数据、异常点排除规则或帕累托判定的前提下，使图在约 3.35 in（85 mm）最终宽度下仍可直接阅读。

## 假设与取舍

- 论文模板尚未指定，采用常见的 3.35 in 单栏宽度；尺寸集中定义在绘图脚本中，后续可按模板微调。
- 三张独立图采用单栏横向画布；原双面板图改为上下堆叠的单栏组合图，避免并排面板被过度压缩。
- 保留 PDF 矢量输出，同时生成 300 dpi PNG 预览。
- 为减少拥挤，uniform baseline 的直接标注使用短名称；完整含义由图例和 README 说明。

## 实施步骤与验证

1. 统一单栏视觉参数
   - 设置画布尺寸、最终印刷字号、线宽、marker、刻度、网格和 PDF 字体嵌入参数。
   - 验证：PDF 页面宽度约为 3.35 in，PNG 宽度约为 1005 px（300 dpi）。

2. 重绘三张独立图
   - 重绘 prefill-only ARC-Challenge、prefill-decode IWSLT 和 CNN/DM 附录图。
   - 缩短坐标轴及点标签，逐图调整标注位置，避免文字、图例和边界互相遮挡。
   - 验证：在最终 3.35 in 显示尺寸下检查标签完整、无裁切且关键点可区分。

3. 重绘组合图
   - 将原左右双面板布局改成上下双面板布局，共享紧凑图例和横轴语义。
   - 验证：组合 PDF 仍为单栏宽度，两面板标题、轴标签和标注均清晰。

4. 回归与记录
   - 重新运行 coverage audit，确认 10/10 uniform baseline 覆盖结论不变。
   - 检查脚本语法、输出文件尺寸、PDF 字体和 PNG/PDF 文件完整性。
   - 在 `dev/impls/131_llama31_single_column_paper_figures_impl.md` 记录修改和验证结果。

## 成功标准

- `paper/` 中四组 PNG/PDF 均被重新生成且宽度符合单栏约束。
- 最小正文/刻度字号在最终尺寸下不低于约 7 pt，标注无明显重叠或裁切。
- 数据点、frontier、uniform baseline 数量及 coverage audit 与修改前一致。
