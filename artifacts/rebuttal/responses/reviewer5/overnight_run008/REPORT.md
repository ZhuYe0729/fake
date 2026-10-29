# Run008：论文呈现、方法定位与历史证据完整性

本目录只处理五位人工reviewer的文档/复核事项，不包含AI Reviewer意见。没有修改原提交论文、共享运行时或已封存001–003结果。

## Figure2可移植版本

从原提交PDF第3页提取原图，保留原图含义和公式，通过原生Ghostscript pdfwrite将字体转轮廓，并导出path-based SVG。输出PDF零字体依赖。288dpi下原图与新图平均像素差0.94/255，任意通道差异超过32的像素占0.694%；已视觉检查公式清晰、内容保留。

原PDF在本机本来即可正常渲染；这一处理消除字体替换依赖，不能声称已经复现reviewer端的全部乱码原因。原论文源文件引用的overview3.pdf不在现有Figures中，因此提供替换图文件，没有猜测重画或改论文源。

- [修正版矢量PDF](results/figure2_outlined.pdf)
- [修正版SVG](results/figure2_outlined.svg)
- [预览PNG](results/figure2_outlined.png)
- [源PDF SHA、工具版本与像素检查](results/figure2_audit.json)

早期CairoSVG mask支持不足、librsvg滤镜分辨率不够的失败版本、日志与代码均保留，正式产物以上述原生pdfwrite版本为准。工具仅在/tmp局部安装/解压。

## NAS背景与相同输入协议

[NAS_CONTEXT.md](NAS_CONTEXT.md)基于ProxylessNAS、FBNet、BRP-NAS原始论文整理定位和英文答复草稿。应承认硬件预测、搜索与资源约束已有先例，将本文贡献准确放在固定LLM的异构格式/阶段分配和运行时实现，避免把通用搜索机制声称为原创。

[IDENTICAL_INPUTS.md](IDENTICAL_INPUTS.md)给出真实归档代码路径与限制：局部候选误差使用相同dense backbone层输入，候选输出不向后传播。原decode局部特征是continuation位置合批M64近似，并非压缩KV的逐步M8重放；策略级NLL评测另行使用真实KV teacher forcing。该近似保持原样并明确披露，不在测试之后重写原拟合。

## 历史证据与共享环境

[只读完整性审计](results/protected_integrity.json)通过：001的15898文件、002的12257文件、003的18372文件，共46527份原归档内容逐SHA校验一致；68份共享运行时源码和编译后的vLLM算子也未改变。原论文SHA为`e4b50838814382c6e6607722a7e591510fc0b029646439ddf06c47a2a6488c3f`。

允许释放的仅为归档目录外、任务拥有且可重建的模型权重二进制，另有逐项cleanup记录；所有代码、日志、CSV、质量结果、文档和失败记录均保留。实验运行时补丁只安装在各自进程，不能把006新增mixed-token变体描述成原提交已经实现的功能。
