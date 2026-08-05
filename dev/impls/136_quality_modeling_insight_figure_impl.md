# 136 Quality Modeling Insight Figure Implementation

## 2026-07-26 - Build paper Figure 3 from frozen single-Linear measurements

- 开发目的：用已有真实 kernel 单 Linear 消融结果完成 quality-modeling insight 图，
  展示 local output error 与整模 \(\Delta\mathrm{NLL}\) 的弱对应关系，以及 depth 和
  projection type 对 sensitivity 的影响。
- 修改内容：新增可复现绘图脚本；筛选 224 条 Llama-2-7B sparse NVFP4 记录；生成
  local-error scatter 和 depth-bucket/type 曲线双子图、PDF/PNG、过滤数据、分组汇总、
  provenance/相关性指标和论文替换 snippet；将 `quality.png` 写入原稿既有 include
  路径。
- 影响文件：`dev/plans/136_quality_modeling_insight_figure_plan.md`、
  `artifacts/debug/070_quality_modeling_insight_figure/`、
  `references/Figures/draft/quality.{png,pdf}` 和
  `references/Figures/draft2/quality_modeling_insight.{png,pdf}`。
- 验证结果：脚本通过 `py_compile`；输入严格为 32 层 × 7 projection types =
  224 个唯一 Linear；scatter Pearson \(r=0.0964\)、Spearman
  \(\rho=0.3540\)；PDF 为单页约 3.33 × 4.45 in，TrueType 字体已嵌入；PNG 为
  996 × 1334；逐图检查无裁切或明显标签重叠。
- 后续注意：`references/wu.tex` 由 `nobody:nogroup` 持有且当前不可写，现有 include
  已能直接加载新图，但 caption/后续句子的收紧版本暂存于 `paper_snippet.tex`。若最终
  需要完全对齐主实验，再按 README TODO 补跑当前模型和四类 fused runtime groups。

## 2026-07-26 - 改为 FakeVLM-style 双栏横向版

- 开发目的：减少第一幅散点图的信息拥挤，使 local error 与 global impact 的趋势不对应
  更直观，同时完整展示各 projection type 在全部 32 层上的 sensitivity。
- 修改内容：panel (a) 改为逐层聚合的 median-normalized mean local error 与 measured
  single-Linear \(\Delta\mathrm{NLL}\) 双曲线；panel (b) 取消 depth bucket，直接绘制
  q/k/v/o/gate/up/down 七类 projection 的 32 层实测曲线；两图改为左右排列，子图标题
  置于各自下方；论文 snippet 改为 `figure*` 和 `\textwidth`。
- 影响文件：`plot_quality_modeling_insight.py`、`README.md`、`paper_snippet.tex`、
  `layer_profile_summary.csv`、全部 Figure 3 PDF/PNG 输出。
- 验证结果：panel (a) layer-profile Pearson \(r=-0.1773\)、Spearman
  \(\rho=0.1580\)；PDF 为单页约 6.77 × 2.68 in，TrueType 字体已嵌入；逐图检查左右
  布局、底部标题、32 层曲线和图例均无明显裁切或重叠。
- 后续注意：当前 `wu.tex` 仍是单栏 `figure` include；应用 `paper_snippet.tex` 后才能
  按设计的双栏宽度排版，直接用现有单栏环境会把横向图缩得过小。

## 2026-07-26 - 收紧子图标题下方排版

- 开发目的：减少横轴与底部 `(a)/(b)` 子图标题之间的多余空白。
- 修改内容：将子图标题锚点由 axes `y=-0.29` 上移至 `y=-0.20`，横轴 label padding
  由 2 降为 0，底部 margin 从 0.24 收紧至 0.21，并将画布高度从 2.65 in 缩至
  2.55 in。
- 影响文件：`plot_quality_modeling_insight.py` 及全部 Figure 3 PDF/PNG 输出。
- 验证结果：PNG 从 2029 × 799 收紧至 2029 × 747；PDF 约为
  6.77 × 2.51 in；底部标题紧邻横轴标签且无重叠、裁切。

## 2026-07-26 - 进一步压缩标题间距

- 开发目的：按反馈将横轴标签与底部子图标题之间的留白压缩到最小安全距离。
- 修改内容：标题锚点继续由 `y=-0.20` 上移至 `y=-0.15`，xlabel padding 调为
  -1，底部 margin 从 0.21 收紧至 0.18，画布高度从 2.55 in 缩至 2.45 in。
- 影响文件：`plot_quality_modeling_insight.py` 及全部 Figure 3 PDF/PNG 输出。
- 验证结果：PNG 进一步缩至 2029 × 717，PDF 约为 6.77 × 2.41 in；标题与横轴
  标签紧凑排列且仍无重叠、裁切。

## 2026-07-27 - 增强单栏缩放后的可读性

- 开发目的：解决横向双子图缩放到论文单栏后字号、线条和标记偏小的问题。
- 修改内容：将源图字号约放大一倍，坐标标签和刻度改为半粗体，子图标题改为粗体，
  同步加粗坐标轴、曲线、网格和标记；长标题折为两行，保持其位于各自子图下方且互不
  重叠。
- 影响文件：`plot_quality_modeling_insight.py`、`README.md` 及全部 Figure 3
  PDF/PNG 输出。
- 验证结果：脚本通过 `py_compile` 并成功重绘；PNG 为 2027 × 878，PDF 为单页约
  6.76 × 2.93 in，TrueType 常规/粗体字体均已嵌入；逐图检查坐标、图例、曲线和两行
  子图标题无裁切或重叠。

## 2026-07-27 - 回调过度放大的图中文字

- 开发目的：修正上一版字号和纵向占用过大、子图标题被迫换行的问题。
- 修改内容：将字号、线宽和标记回调为相对原版约 20–30% 的温和增强，保留标签半粗体
  和标题粗体；两个子图标题恢复单行，并缩小画布高度、轴间距和底部留白。
- 影响文件：`plot_quality_modeling_insight.py`、`README.md` 及全部 Figure 3
  PDF/PNG 输出。
- 验证结果：脚本通过 `py_compile`；PNG 为 2051 × 746，PDF 为单页约
  6.84 × 2.50 in，常规/粗体 TrueType 字体均已嵌入；视觉检查确认两侧纵轴标签均位于
  图像边界内，两个底部标题保持单行且无重叠。

## 2026-07-27 - PRO 6000 原始 BF16 WikiText 临时诊断

- 开发目的：绕过 COSPAQ exporter，区分 Llama3 跨平台质量差异来自软件栈还是导出
  checkpoint/runtime。
- 修改内容：新增单文件脚本，自动加载 067 的 PRO 6000 环境并直接评测原始 BF16
  模型的 WikiText PPL。
- 影响文件：`scripts/check_pro6000_raw_bf16_wikitext.sh`
- 后续注意：该脚本仅用于紧急诊断，结果确认后可删除。

## 2026-07-27 - 凝练 Figure 3 标题并统一面板宽度

- 开发目的：提高图中文字可读性，并修正相同 layer 范围在左右子图中显示长度不一致的
  视觉问题。
- 修改内容：将全局、坐标、刻度、图例和子图标题字号温和放大约 10–13%；标题简化为
  `Local error fails to predict impact` 和 `Sensitivity depends on depth and type`；
  将原先人为设置的 0.95:1.45 面板宽度改为 1:1，并共享 layer 0–31 横轴。
- 影响文件：`plot_quality_modeling_insight.py`、`README.md` 及全部 Figure 3
  PDF/PNG 输出。
- 验证结果：脚本通过 `py_compile`；左右绘图区等宽且横轴范围、刻度完全一致；PDF
  为单页约 6.88 × 2.50 in，常规/粗体 TrueType 字体均已嵌入；视觉检查确认标题保持
  单行，七类 projection 图例、双纵轴标签和全部数据无裁切或重叠。

## 2026-07-27 - 简化右侧纵轴标题

- 修改内容：将右侧纵轴从 `Measured single-Linear ΔNLL` 简化为
  `Measured ΔNLL`，其余数据和排版保持不变。
- 影响文件：绘图脚本及全部 Figure 3 PDF/PNG 输出。

## 2026-07-28 - 修正主结果表的 Llama-3.1 Balanced 选点

- 开发目的：避免将偏速度优先且 MMLU 明显下降的 `point_009` 表述为紧质量预算下的
  Balanced 策略。
- 修改内容：新增可直接通过 `\input{}` 引用的主结果表；Llama-3.1 prefill-only
  Balanced 改用 `point_007`，prefill--decode Balanced 保持 `point_005`。
- 影响文件：`artifacts/debug/071_my_emergency/main_table.tex`。
- 验证结果：表格各数据行均为 13 列；更新后的 Llama-3.1 Balanced 为
  ARC-C 51.45、ARC-E 79.08、WinoGrande 73.80、MMLU 66.50、TTFT 1.59×。
