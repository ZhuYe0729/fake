# Run002 调度与恢复记录

2026-10-25：以下为实验过程记录。当前全部完成，所有最终工作队列正常退出。

- 三个标准 cal role 绑定 GPU0/4/7；每个 role 新增 16 个 prefill + 16 个 decode 校准标签。父队列 PID 初始分别为 1392530、1392264、1392543。
- `residual_queue.py --slot 0/1/2` 等待对应校准完成标记后，在空闲边界暂存父队列，分别完成 SQ-INT8 / FP8 / SQ-FP8 完整 baseline，再恢复原父队列做分配策略质量评测。只操作本实验父进程。
- 初始 GPU3 primary 在六个初始速度点和 BF16 NLL 后，于无子进程的等待阶段被重新加载。已测 SmoothQuant-INT8 质量的初次 tasks 因 Hub offline 失败；原 worker 继续三项 generation 后按其失败记录退出，退出码 1 属于保留的已知旧尝试。
- 第二个 baseline 等待者尚未获得 policy lock，已终止并记录到 `results/queue_reschedule.json`，避免三种 baseline 在 GPU3 串行阻塞测速。
- `residual_queue.py --slot primary` 绑定 GPU3，等待初次 SQ-INT8 generation 完整结束且原协调器退出，再执行更新的 `experiment_queue.py --role primary`。
- 更新的 primary：补测剩余旧 uniform 基线速度；先测已冻结的 prefill 策略，与 decode 校准并行；等全部校准完成后拟合 decode、发布完整 evaluation_points，并测余下所有策略速度。
- GPU3 的每个分配策略测速结束后释放自有可重建模型二进制，再写 speed_complete 标记。质量 role 之后重新导出，防止同时导出/释放同一文件。
- 质量 role 初始按 evaluation_points 索引 modulo 3 分工；完整任务耗时差异使负载不均。后续在现有完整策略子进程结束后，`switch_quality_queue.py` 停用旧的等待父队列并 exec 到共享 `quality_queue.py`，所有代码、日志与已完成结果保留。
- 共享队列以每个 policy 的非阻塞 claim lock 分配独立任务，先补齐各点 NLL，再做预先指定的完整任务；同一 policy 的两个阶段共用锁，释放模型后再交接。数据、策略、指标与选择规则没有变化。
- GPU3 全部速度完成后，`phase_quality_helper.py` 用该卡提前完成两个固定 8bit+Marlin phase 控制的 NLL/生成任务。与后续标准队列的 policy lock 共用，已完成结果会跳过；progress 额外检查 helper complete。
- 统一速度完成后，按 TTFT/E2E 速度选每组最快点，原预算代表点均保留；唯一新增完整任务为 `daso8_prefill_decode_01`。`supplement_fastest.py` 在 GPU3 当前完整策略结束后临时接手该卡补测，再恢复共享队列。选择未使用 NLL 或任务分数，单独配置记录。

最终收尾顺序：`report_results.py` 无 missing → `--validate-final` 生成数表 → `plot_results.py` → `write_report.py` → `audit_8bit.py` → 更新 README / impl → `report_results.py --validate-final` 重建最终 manifest → `package_responses.py` → 校验证据包哈希及链接。最后一次修改 run002 之后必须重建 manifest 与证据包 source hash。

001 只读：新数据加载禁用 datasets 缓存，test/validation 直接读取归档 Arrow，初始化用的匹配 train/dev 另存到 run002。无 001 新增未列入原 manifest 的文件。

## 最终状态
- 47 个速度配置 / 57 个测试 NLL 点 / 80 项 likelihood / 57 份生成结果完整，新增生成 33,996 个样本。
- 最终四个共享质量 worker 和追加最快点作业均退出码 0，GPU 测试进程已结束。
- 完整性审计通过；001 的 15,898 个文件和 60 个复用源文件哈希保持一致。
