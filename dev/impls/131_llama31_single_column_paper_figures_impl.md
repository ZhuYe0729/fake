# 131 Llama3.1 single-column paper figures implementation

## 2026-07-23 - Redraw all paper figures for one-column placement

- 开发目的：将 debug 068 `report/paper/` 中的三张独立图和一张双面板组合图改为论文单栏最终尺寸，并保证文字与标注可读。
- 修改内容：统一使用 3.35 in 宽画布、7--8 pt 最终字号、紧凑线宽和 marker、短 baseline 标签、300 dpi PNG 与嵌入 TrueType 字体的矢量 PDF；三张独立图为 3.35 × 2.45 in；组合图由左右排列改为上下排列并收紧为 3.35 × 4.45 in，上面板省略重复横轴标题。
- 影响文件：`build_two_scenario_paper_figures.py`、`llama31_8b_instruct/report/paper/` 下全部 PNG/PDF、README 和 coverage audit。
- 验证结果：脚本通过 `py_compile`；四个 PDF 页面宽度均为 241.2 pt（3.35 in），PNG 宽度均为 1005 px；PDF 字体已嵌入；prefill/decode frontier 分别保持 13/7 个点，uniform baseline 覆盖仍为 10/10；逐张视觉检查未发现裁切或明显标注重叠。
- 后续注意：若论文模板的实际单栏宽度不是 3.35 in，只需修改脚本顶部 `SINGLE_COLUMN_WIDTH`，不要在 LaTeX 中先按大图生成再大幅缩放。

## 2026-07-23 - Add side-by-side single-column alternative

- 开发目的：提供一版可直接与上下双面板版本比较的横向单栏组合图。
- 修改内容：新增 `llama31_two_scenario_tradeoff_horizontal.{png,pdf}`；保持 3.35 in 单栏宽度，将两个面板左右排列，共享图例和横轴标题，减少主刻度数量并使用紧凑标注。
- 验证结果：横向 PDF 页面为 241.2 × 140.4 pt（3.35 × 1.95 in），PNG 为 1005 × 585；TrueType 字体已嵌入；脚本通过 `py_compile`；视觉检查中标签完整且无裁切。
- 后续注意：横向版最终文字约 5--6 pt，适合优先节省纵向空间；若模板或审稿要求最小 7 pt，应使用原有的上下双面板版本。

## 2026-07-23 - Replace horizontal in-plot callouts with legend encoding

- 开发目的：移除横向单栏版中拥挤的 `Bal.` 和 `Max` 直接标注，同时保留两个代表性策略点的语义。
- 修改内容：Balanced 改为蓝色菱形、Max speed 改为橙色星形；两类 marker 在两个面板保持一致，并与 Frontier、Uniform 一起进入顶部四项共享图例；上下双面板版不变。
- 验证结果：四项图例仍能在 3.35 in 宽度内单行完整显示；数据区不再出现 Balanced/Max 文字；PNG/PDF 无裁切，页面尺寸和 coverage 结论不变。

## 2026-07-27 - Enlarge horizontal trade-off figure typography

- 开发目的：提升横向双面板图在论文单栏中的文字可读性。
- 修改内容：仅放大横向版本的正文、坐标轴、标题、刻度、图例和直接标注字体，最终字号约为 5.4--7.1 pt；画布高度由 1.95 in 微调至 2.05 in，并错开 decode 面板中的 BF16/W4A16 标注以避免重叠；实验数据与高亮策略不变。
- 影响文件：`build_two_scenario_paper_figures.py`、`llama31_two_scenario_tradeoff_horizontal.{png,pdf}`。
- 验证结果：脚本通过 `py_compile`；PDF 为单页 3.35 × 2.05 in，PNG 为 1005 × 615；TrueType 字体已嵌入；coverage audit 仍为 10/10；视觉检查未发现图例越界、文字裁切或标注重叠。
