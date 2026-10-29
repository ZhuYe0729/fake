# Run008 — 人工reviewer文档与图形补充

原reference论文保持原样。Figure2在本机可以正确渲染，原字体已嵌入；另提供字形全部转为路径的矢量PDF/SVG，消除字体替换依赖。未改公式语义，未声称复现所有reviewer端显示问题。

[无字体依赖PDF](results/figure2_outlined.pdf) · [SVG](results/figure2_outlined.svg) · [渲染图](results/figure2_outlined.png) · [审计](results/figure2_audit.json)。

重现：pypdf工具装在`/tmp/daso_rebuttal_vector_tools`，Ghostscript Debian包解压在`/tmp/daso_rebuttal_ghostscript`，没有全局安装；执行code/outline_figure2.py，使用pdftocairo/pdftoppm/pdffonts。SVG转PDF的mask/低分辨率失败样本及脚本保留。
