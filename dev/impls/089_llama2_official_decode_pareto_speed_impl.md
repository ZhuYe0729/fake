## 2026-07-12 - Runner discrepancy reproduction
- 开发目的：确认 prefill-decode max-speed 的 1.65x 是否为可复现的正式口径。
- 验证结果：当前 Pareto point 11 与旧 max-speed checkpoint 的逐 module、逐 phase policy 完全一致。以旧 `benchmark_phase_hetero.py`、`max_model_len=2128`、`.9` 显存与旧 phase-runtime 环境执行，得到 2974.807 ms；相对 4868.068 ms dense-BF16 基线为 1.636x，复现旧结果。
- 发现：此前 provisional 图采用 `benchmark_one.py`、`max_model_len=4096`、prefix cache 和 `.8` 配置，给出 1.40x，不能用作正式 decode 图。point 8 在 `.9` 下稳定 OOM（NVFP4 prefill workspace 额外需要约 1.34 GiB），须显式处理。
- 后续：按 plan 089 使用旧正式 runner 重测并重建最终 decode 曲线。

## 2026-07-12 - Official runner speed remeasurement launch
- 开发目的：开始替换 provisional `.8` decode 数据，恢复此前正式 max-speed 的速度口径。
- 修改内容：新增 `run_official_decode_speed.sh`；独立写入 `validation/prefill_decode/speed_official/`，固定旧 phase-runtime 环境、`max_model_len=2128`（由旧 runner 按场景设置）、`.9` 显存、output=1/80 各 1 warmup + 10 次 fresh-process。
- 验证：shell 语法与 diff 检查通过；已在 GPU 7 启动 point 0、3、6、11 的串行重测。
- 后续注意：point 8 在正式 `.9` 配置 OOM，不能进入该批；待测试 point 7 或其他邻近替代点。

## 2026-07-12 - Official decode figure rebuild
- 开发目的：以正式 10 次 fresh-process 中位数替换图中的 `.8` 临时 decode 速度。
- 修改内容：新增 `build_official_decode_comparison.py`，输出 `measured_comparison_official.csv`；报告与 speedup 图改读该表。正式 decode 图仅连结实际全局 Pareto 的 point 6、11，并保留 dense-BF16 的历史正式基线；point 0/3 在表中透明保留但不连为前沿，point 8 标为正式 `.9` 配置 OOM。
- 验证：重建 CSV、两张 speedup 图和总览 `measured_pareto.png` 均成功。point 11 为 3039.784 ms / 1.602x。
