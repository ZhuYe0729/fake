# Llama3.1 ablation consolidation implementation log

## 2026-07-22 - Complete ablation artifact and paper section

- Consolidated experiments 064--066 into
  `artifacts/debug/067_llama31_ablation_consolidation/` without modifying any
  source or main-result artifact.
- Copied 140 authoritative files (about 3 MB): raw prediction rows, metrics,
  fitted quality models, calibration repeats/extensions/policies/logs, runtime
  paired samples/logs, three CUDA profiles, figures, LaTeX tables, detailed
  documentation, and reproduction scripts.
- Excluded the reproducible 12 GB runtime checkpoint while retaining its solved
  policy and canonical-export provenance.
- Added `source_manifest.json` with source/destination paths, sizes, and SHA256,
  plus `data/key_metrics.csv` for the 13 paper-facing values.
- Added a detailed Chinese `summary.md` and an English, ablation-only
  `ablation.tex` following the DASO terminology and organization in
  `references/wu.tex`. The paper fragment uses a compact modeling table, the
  calibration-size curve, and a separate E2E/direct-overlap runtime table.
- Added `scripts/validate.py`; it verifies all copied hashes, expected metrics,
  TeX figures, measured-result literals, placeholder absence, and checkpoint
  exclusion. Validation passes. No LaTeX engine is installed on this host, so
  full PDF compilation remains a portability check for the paper environment.

## 2026-07-27 - 美化 calibration-size sensitivity 图

- 开发目的：增强论文单栏缩放后 calibration-size sensitivity 图中文字的可读性，同时
  避免字号或纵轴占用过度。
- 修改内容：在 065 的可复现绘图脚本中将正文、刻度和图例字号温和提高约 10–15%，
  坐标标题使用半粗体，适度加粗两条曲线并放大标记；将原四行透明图例整理为顶部两列
  带轻微白底的图例，避免右上角成本曲线穿过文字；重新生成并同步 067 的论文版和完整
  审计版 PDF/PNG。
- 影响文件：065 `scripts/build_report.py` 与四个 report 图，067 的四个 figure 图、
  `reproduction/calibration/build_report.py` 和 `source_manifest.json`。
- 验证结果：065 与 067 的论文版 PDF/PNG 哈希分别一致；067 `validate.py` 全部通过
  （140 个 provenance 文件、13 个关键指标及 TeX 图引用）；PDF 为单页约
  7.47 × 4.28 in，常规和粗体 TrueType 字体均已嵌入，视觉检查无标签裁切或图例遮挡。

## 2026-07-27 - 再次放大 calibration-size sensitivity 图中文字

- 开发目的：解决大画布图缩放到论文单栏后有效字号仍偏小的问题。
- 修改内容：将源画布从 7.6 × 4.6 in 调整为 5.6 × 3.8 in，同时将正文、坐标标签、
  刻度、图例和标注字号进一步提高；左轴简化为 `ΔNLL prediction MAE`，右轴简化为
  `Calibration cost (GPU-min)`，星标图例简化为 `Selected setting`；同步调整图例和
  完整审计图标注位置。
- 影响文件：065 `scripts/build_report.py` 与四个 report 图，067 的四个 figure 图、
  `reproduction/calibration/build_report.py` 和 `source_manifest.json`。
- 验证结果：绘图脚本通过 `py_compile`，067 `validate.py` 全部通过；论文版 PDF 为
  单页约 5.38 × 3.36 in，按 3.35 in 单栏缩放后刻度约 7.5 pt、坐标标签约 8.4 pt；
  常规和粗体 TrueType 字体均已嵌入，视觉检查无标签裁切、图例遮挡或文字换行。

## 2026-07-27 - 收紧 calibration-size sensitivity 图例

- 开发目的：修正放大后的图例接近右侧边界并与右轴刻度重叠的问题。
- 修改内容：仅将图例字号从 11 pt 调整为 10 pt，并缩短图例线段、收紧列距、行距和
  内边距；坐标标签、刻度和数据区域保持不变。
- 影响文件：065 绘图脚本与四个 report 图，067 对应 figure、reproduction 脚本和
  provenance manifest。
- 验证结果：067 完整验证通过；视觉检查确认四项图例均位于图框内，与右轴刻度和曲线
  无重叠。
