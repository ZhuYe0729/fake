# Qwen3 Table 2 completion plan

## 目标与固定范围
- 补齐 `artifacts/rebuttal/responses/reviewer2/qwen3_table2/README.md` 现有17行、13列，不根据新结果换策略或删除方法。
- 新建隔离 run013；原001–012、原始日志和失败记录保持不变，旧主表保存快照。
- 精度继续使用原完整任务协议：ARC-C acc_norm、ARC-E acc、Winogrande/MMLU acc；CNN/DM1000、DialogSum1500、IWSLT333。已有相同map/权重/后端的完整任务结果继承并核验来源，缺失的3个prefill map和2个P+D map实际补测。

## 步骤与验证
1. 冻结现有表格到唯一map与质量来源的映射，保存代码、输入、后端与旧manifest SHA。
   - 验证：每个半行明确配置；不得使用旧HAWQ分数代填新map。
2. 空闲GPU并行完成缺失完整质量任务，保留逐样本生成答案和原始lm-eval结果。
   - 验证：任务样本数完整、limit=None、生成输入SHA一致；未读这些分数选配置。
3. 其他本轮GPU任务结束后，GPU0/node0/CPU0–7完成三轮独立进程同卡测速。
   - B8/S2048/O1或64，6GiB BF16 KV，eager、无prefix/chunked，3 warmup+8 timed；冻结相同输入seed144983，方法顺序三轮预先固定，BF16每轮首尾锚点。
   - Prefill用O1完整生成时间；P+D在EngineCore输出进入FINAL_ONLY过滤前观测实际首token，按同次 `(E2E-TTFT)/63` 计算TPOT；记录阶段恢复与完整周期，恢复是否计入与原协议一致。
   - 验证：完整32个policy/scenario组合均覆盖三轮，输出token计数与输入一致，首token早于完成，逐轮结果和范围同时报告，不选最好一轮。
4. 从原始文件生成主表/CSV/LaTeX及逐项来源；追加开发记录与完成审计。
   - 验证：17行187个指标全部有限且无待补；速度均来自新统一campaign，完整policy signature与数据SHA可追溯。

## 资源
- 模型导出放任务专属 `/dev/shm/daso_rebuttal_013_20261026`，每次用后可删除可重建二进制，保留policy、命令、metadata及结果。
- 如空间不足，仅清理已归档且无进程使用的任务自有可重建checkpoint二进制，逐项记录；不删除canonical稀疏权重、代码、日志、CSV、文档或结果。
