## 2026-06-01 - Qwen3.5 五种真实 kernel 路径
- 开发目的：补齐 Qwen3.5 的 dense BF16、dense NVFP4、sparse BF16、sparse NVFP4、Marlin W4A16 五种真实端到端速度路径。
- 修改内容：新增 Qwen3.5 packed kernel checkpoint helper；新增统一 prepare 脚本；扩展 Qwen3.5 speed benchmark 的 `--method` 支持五种真实路径，压缩路径 runtime 只加载 packed checkpoint。
- 影响文件：`fake/models/qwen3_5_kernels.py`、`scripts/prepare_qwen3_5_kernel_checkpoint.py`、`scripts/bench_qwen3_5_speed.py`。
- 后续注意：当前会话 CUDA 不可用，未执行 GPU prepare/forward；已完成 py_compile 和 CLI help 检查。

## 2026-06-01 - Qwen3.5 其他尺寸并行测试命令
- 开发目的：整理 Qwen3.5 2B/4B/9B/27B 在五种真实 kernel 路径上的 prepare 与 speed benchmark 命令，便于多 tmux 多 GPU 并行执行。
- 修改内容：新增根目录临时命令文档，按 card 1 起分配 GPU，并扩展 benchmark grid 到 batch size `1 2 4 8 16 32` 与 input tokens `128 512 1024 2048 8192`。
- 影响文件：`tmp.md`、`dev/impls/033_qwen35_real_kernel_paths_impl.md`。
- 后续注意：当前脚本未实现 27B 的 tensor parallel/device_map；`CUDA_VISIBLE_DEVICES=4,5` 只保证两张卡对进程可见。

## 2026-06-01 - 本机环境命令调整
- 开发目的：将 Qwen3.5 并行测试命令从超算环境调整为本机直接运行。
- 修改内容：移除 `module load cuda/12.8`、超算 conda 初始化和固定 `HF_HOME`；conda 环境改为 `cospaq`。
- 影响文件：`tmp.md`、`dev/impls/033_qwen35_real_kernel_paths_impl.md`。
- 后续注意：仍保留 `CUDA_VISIBLE_DEVICES` 按可见 GPU card 分配。

## 2026-06-01 - 27B 多卡分片加载
- 开发目的：修复 27B 即使暴露多张 GPU 仍因 `.to("cuda")` 整模放入可见 `cuda:0` 而 OOM 的问题。
- 修改内容：prepare/benchmark 增加 `--device-map` 和 `--max-memory`；compressed checkpoint 恢复支持 `device=auto`，按原 sharded module 所在设备放置 packed 权重；27B 临时命令改为 card `4,5,6,7` 加 `--device-map auto`。
- 影响文件：`scripts/prepare_qwen3_5_kernel_checkpoint.py`、`scripts/bench_qwen3_5_speed.py`、`fake/models/qwen3_5_kernels.py`、`fake/kernels/marlin_nvfp4.py`、`tmp.md`、`dev/impls/033_qwen35_real_kernel_paths_impl.md`。
- 后续注意：`--max-memory` 的 GPU 编号是 `CUDA_VISIBLE_DEVICES` 后的相对编号。
