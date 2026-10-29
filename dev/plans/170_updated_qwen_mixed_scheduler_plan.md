# 170 — Qwen新点真实混合调度补充（R2 Exi5 / R1 AK6a / R5 gkP9）

## 目标与边界
- 在原006已验证的实际vLLM V1 chunked-prefill / mixed-token双表示驻留变体中，测试040预先固定的同九个map，尤其032/033两个新DASO探索点。不同于主表全局同阶段单表示/arena，明确报告dual-resident的GPU/host内存；不冒充生产open-loop服务。
- 只选两个006原代表case：B8/S2048/O64与B8/交替S512、2048/O256，共同chunk预算1024。所有请求t0进入，固定原speed seed144907；不新增逐负载求解或按结果选配置。
- 数值核使用040与033修正后的算子文件；不修改006/040及封存主表。候选BF16、INT8、SQ-INT8、FP8、SQ-prefill/W4A16-decode、两个HAWQ/硬件端点、032 q=.01和033 wP32/q=.03，全部保留。

## 实现与验收
1. 创建独立047目录，复制已验证的数值核、006 resident/mixed routing及teacher forcing；冻结源SHA、九map、两个case和数据来源。验收：设计先于本run任何GPU输出；代码语法和输入64窗口/分组CPU检查。
2. 使用019原long_quality 64个已查看窗口、同case实际prompt长度/输出长度，所有方法强制同目标tokens。先测BF16 routed/no-op，检查每窗口NLL一致；每种跨格式模块执行精确gather/apply/scatter审计，检查真实mixed forward、阶段token总数和异格式Linear调用数。验收：64窗口完整、finite logprob、所有目标逐token一致；ΔNLL保护值mean+2SE+.005仅描述，不宣称名义q通过。
3. 固定同一GPU/NUMA、KV6GiB/eager/no prefix cache，三轮每轮2warmup+5timed，两case同进程；每轮随机中间顺序、首尾BF16。质量SH、速度EX与其他实验互斥。只使用空闲卡，不中止其他进程。验收：每方法三轮/每case15个timed burst、首尾漂移、同轮BF16/INT8归一化、每请求p50/p95/p99和失败/内存等完整记录。
4. 完成后交付reviewer2/qwen_updated_mixed_run047的完整表、CSV、实际路由/内存、来源和审计；保留所有失败与不利排名，不由局部case推广为全局最优。只清理本run可重建model.safetensors，代码/文档/日志/CSV/原始captures保留。

## 调度
- 当前034–046优先。047等待三模型新窗口复核均完成后开始，避免干扰主表/反约束/质量预算核心证据；若资源或时间不足，如实保留待完成状态。
- 无HTTP、网络或随机到达率/生产SLA声明；不将两个closed-loop burst代替open-loop生产评测。
