# 153 — DASO新实验前清理已完成Llama的可重建导出缓存

## 授权与边界
- 用户明确要求在开始新测试前先清理基本不再需要、可复现的大文件；任何代码、文档、日志、CSV及其他实验结果均保留。
- 本次仅删除两个完成Llama实验的41份 `night151/primary_weight_cache/{017,018}/*.safetensors`，以及022完成预算测速的4份 `data/checkpoints/*/*/model.safetensors`，预计约373 GiB。
- 不删除任何目录、原始模型、数据集/校准数据、canonical剪枝状态、Qwen近期使用的027权重、编译kernel、其他任务临时文件或结果。

## 执行与验收
1. 冻结45个文件白名单、字节数、inode/stat、重建policy与export命令、已记录SHA及header SHA；确认017/018/022最终审计完成。
2. 通过宿主实际文件系统核验原始模型和canonical依赖仍在；检查目标文件未被进程打开或mmap。使用measurement/export/allocation锁避免与遵循现有协议的实验竞争，不停止任何外部进程。
3. 删除前后核验rebuttal区域所有非目标代码、文档、日志、结果及配置内容哈希；大型非目标二进制保留并核验stat。只逐文件unlink白名单，保留缓存引用symlink及全部元数据。
4. 输出清理清单、重建命令、实际回收空间及保护审计；注明旧缓存symlink可能指向已清理的权重，重建即可恢复，原结果未删除。

## 记录
- 交付位于 `artifacts/rebuttal/cleanup/20261027_pre_daso_optimization/`；开发记录追加impl153。
- 已有权重SHA使用之前复制校验记录，另记录当前stat/header SHA，不冒称重新流式核验了数百GiB的每个旧权重。022没有旧整文件SHA时现场计算。
- 清理完成前不开始新DASO GPU实验。
