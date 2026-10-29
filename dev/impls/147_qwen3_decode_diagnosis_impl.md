## 2026-10-26 - 启动 TPOT 诊断
- 目的：确认 W4A16 算子收益未反映到 run013 TPOT 的具体原因。
- 新建计划147与隔离 run014；沿用固定输入和同卡/同 NUMA 协议，保留原实验与主表。
- 初步证据：uniform W4A16 的 host 权重为零且前后阶段同方法，不支持“逐 token H2D 搬运”解释；当前正式测速为 eager。
- 后续：先复现和分解，再决定是否修复/扩大重测，不把共享开销直接当作倒退原因。

## 2026-10-26 - 完成受控定位和原生 Graph 验证
- 原因：固定真实 KV 的完整模型 GPU kernel 总时间约 BF16 12.66ms、NVFP4 W4A16 6.41ms、INT4 W4A16 6.33ms；eager 的 CPU 调用/发射开销掩盖收益。INT4 通用绑定有较多临时/零长度张量操作；未将 aten::empty 次数等同 cudaMalloc。
- 实测：同进程 ABBA/BAAB、每模式8次正常 B8/S2048/O64 生成。TPOT eager→Graph：BF16 16.867→13.517ms、NVFP4 15.304→7.301ms、INT4 18.124→7.147ms。均启用 Graph 后，两种 W4A16 相对 BF16 为1.851×/1.891×。
- 验证：另有18进程两轮倒序控制；三模型冻结 positions=2110/seq_lens=2111 的 Graph logits 最大绝对差均0，BF16/NVFP4 同进程完整生成 tokens 一致。INT4 已知生成变化保留，不据输出筛样本。
- 影响文件：run014 全部代码/配置/trace/CSV/报告；reviewer2/qwen3_table2 新增 TPOT_DIAGNOSIS.md，README 与生成器补诊断链接，主表 CSV/LaTeX/provenance 数值不变。
- 隔离与审计：未修改共享 vLLM 或 CUDA kernel；原6份后端代码拷贝 SHA 一致，编译 ops SHA 一致，run013 manifest 中1165项全部核验。GPU0已释放；任务导出放专属 /dev/shm，未清理任何既有文件。
- 恢复记录：首次后处理审计把 symlink manifest 条目当普通文件，触发 KeyError；仅修复 manifest schema 处理并重跑分析。29个命令中唯一非零退出为该后处理，原日志/代码快照与恢复记录保留，全部24个正式测量进程成功。
- 边界：CPU频率与TPOT共同波动为关联证据，未修改全局时钟；uniform 可直接冻结阶段并使用原生 Graph。DASO/HAWQ 的权重生命周期尚需统一处理，不能把本轮 Graph 基线与旧 eager 异构策略混排，当前不据此更新算法排名。
