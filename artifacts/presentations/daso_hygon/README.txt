DASO 技术交流（5 页修订稿）

直接使用
  DASO_技术交流.pptx：可编辑的 16:9 演示文稿，附逐页讲述备注。
  DASO_技术交流.pdf：由同一排版源生成的矢量预览，已嵌入中文字体。
  previews/：逐页 PNG 与 overview.png 总览。

页面逻辑
  1. 背景：从压缩方法到部署决策。
  2. 观察：Prefill / Decode 的方法收益不同。
  3. 方法（一）：硬件与形状感知的时延模型，以及结构和阶段感知的质量模型。
  4. 方法（二）：质量预算约束下的策略优化，以及 vLLM 阶段异构运行时。
  5. 结果：论文原始双面板质量—速度图，以及最终主表的主要性能结果。

口径
  以 artifacts/paper/wu.tex 与 daso/results/paper/ 的最终版本为准。
  2.26× Prefill 来自 Llama-3.1，1.55× E2E 来自 Llama-2，硬件均为 RTX 5090。
  上述最大加速对应宽松质量预算，不表示无损；Balanced 是较严格预算的代表点。
  第 3 页使用质量建模原图，说明局部误差与整体质量影响的区别。
  第 5 页原样嵌入论文的 Llama-3.1 双面板图，包含 prefill-only ARC-C
  与 prefill-decode IWSLT SacreBLEU；保持原图的坐标、图例和数值。
  结果页右侧数值来自两模型主表，并明确区分最大加速所对应的模型。
  本次修订删除了合作展望，内容集中于已完成工作的技术交流。

材料来源
  artifacts/paper/wu.tex
  daso/results/paper/tables/main_table_final.tex
  daso/results/paper/tables/main_table_sources.csv
  daso/results/paper/tables/cross_platform_pro6000_final.tex
  daso/results/paper/figures/llama31_two_scenario_tradeoff_horizontal.png
  daso/results/paper/figures/quality_modeling_insight.png
  assets/data.json 保存本次图表所用数值快照；原始论文与归档保持原样。
  assets/paper_figures/ 保存原样复制的论文图片，PPT 与 PDF 均直接嵌入。

修改与重建
  安装依赖：python -m pip install python-pptx reportlab
  在本目录执行：python build_deck.py
  若需更新 PNG 预览：pdftoppm -r 110 -png DASO_技术交流.pdf previews/slide
  layout_check.json 是生成时的文字框与字号检查记录。

字体
  PPT 使用 Noto Sans SC；本机缺少时可安装完整字体或替换为思源黑体 / 微软雅黑。
  assets/fonts/ 中为重建和 PDF 嵌入所需的字体子集，许可证一并保留。
  来源：https://github.com/notofonts/noto-cjk

本材料为技术交流初稿，未加入个人、单位署名或论文录用状态。
