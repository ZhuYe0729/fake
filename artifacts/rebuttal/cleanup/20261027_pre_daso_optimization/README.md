# DASO新实验前的可重建权重缓存清理

2026-10-27完成。按用户授权，仅删除45份已完成Llama实验的派生权重，合计373.286 GiB：017的21份primary cache、018的20份primary cache、022的4份补充测速导出。原始模型、canonical剪枝状态、Qwen近期需要的027权重及校准数据全部保留。

清理前核验实际宿主文件系统的模型/剪枝重建依赖，并扫描所有进程fd和mmap：目标文件0占用、0不可读进程。持有现有measurement/export/allocation锁，不停止任何进程。

删除前后118,423份非目标文件保护核验一致，代码、文档、日志、CSV、配置、原始结果、symlink文本均未改动；未删除任何目录。审计快照的磁盘可用空间由172.27 GiB增至545.06 GiB，实际字节数见审计。

- [45份文件与重建命令](manifest.json)：保存policy/source/canonical依赖、字节数、inode、mtime、header SHA及已有或现场计算的文件SHA。
- [逐文件删除记录](deleted.jsonl)
- [完成和保护审计](completed.json)
- [删除前保护指纹](protected_before.json)
- [进程占用检查](process_check.json)
- [清理脚本](cleanup.py)

原primary cache的文件SHA来自之前复制后逐字节校验记录，本轮另检查stat和header；022四份权重在本轮流式计算SHA。没有将旧记录冒充本轮重新计算全部373 GiB的完整SHA。

部分旧`/dev/shm/.../model.safetensors` symlink仍指向已清理的cache路径，属于有意驱逐的可重建缓存，symlink和其他元数据均保留。需要复跑时按manifest里的`rebuild_command`导出到新空目录，再仅复制`model.safetensors`恢复对应路径。**不要使用exporter的`--force`覆盖旧目录**，以免删除已保留的元数据。017/018原checkpoint helper也会在缺少权重时保留旧目录后重新导出。
