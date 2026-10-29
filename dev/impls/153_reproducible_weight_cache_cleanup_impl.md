## 2026-10-27 — 清理已完成Llama的派生权重缓存
- 用户要求新测试前先清理可重建大文件；本轮未启动新的GPU测试。
- 删除固定45份safetensors：017 primary cache21份、01820份、022补充测速4份，共373.286 GiB。仅逐文件unlink，不删目录；保留全部代码、文档、日志、CSV、配置、原始结果、symlink及canonical/source数据。
- 先核验完成审计、policy SHA、原始模型/canonical依赖、缓存stat/header与重建命令；现场流式计算022四份SHA，其余复用之前复制校验的文件SHA。原exporter拒绝已有目录，重建命令改为导出新空目录，不建议会删元数据的`--force`。
- 宿主全部进程fd/mmap目标引用0、不可读进程0；持有measurement/export/allocation锁，未停止外部进程。删除前后118423份非目标文件保护指纹一致，missing/changed均0。
- 项目盘审计快照可用空间由172.27 GiB增至545.06 GiB；Qwen027当前权重、原始模型、canonical剪枝状态、校准数据和共享kernel保留。旧primary cache symlink可暂时悬空，原引用文本保留，重建后恢复。
- 交付：`artifacts/rebuttal/cleanup/20261027_pre_daso_optimization/README.md`、manifest、deleted journal、completed audit及脚本。后续DASO优化另立plan，避免将缓存清理与方法改动混在一起。
