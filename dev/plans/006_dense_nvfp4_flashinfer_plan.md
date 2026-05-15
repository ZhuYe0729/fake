# 006 Dense NVFP4 FlashInfer Kernel Plan

## 目标

先在 MaxViT 上实现 dense NVFP4 的真实 kernel 推理路径，用 FlashInfer 的激活 NVFP4 量化和 FP4 GEMM 替换当前 fake-quant 后仍以普通 dense Linear/Conv 执行的路径，最终产出 dense MaxViT 与 NVFP4 MaxViT 的端到端 forward speed 对比。

范围先保持收敛：
- 模型：MaxViT，优先 tiny，后续复用到 small/base/large。
- 压缩形态：dense NVFP4，不叠加 sparsity。
- 模块类型：第一阶段只替换 `nn.Linear`；第二阶段评估 MaxViT MBConv 中可等价展开为 GEMM 的 1x1 `Conv2d`。
- 目标设备：RTX 5090 / Blackwell。FlashInfer 运行脚本允许切换 `module load cuda/12.9`。

## 现状判断

当前 `fake.compression.nvfp4` 是 fake quant：权重量化后反量化回原 dtype，再写回 `module.weight`，测速时仍走 PyTorch/timm 原始 dense kernel。它能评估近似精度损失，但不能体现 FP4 Tensor Core 推理加速。

现有 MaxViT 入口和测速脚本已经比较适合复用：
- `fake.models.maxvit.load_maxvit_dense`
- `fake.compression.modules.select_compressible_modules`
- `scripts/prepare_compressed_model.py`
- `scripts/bench_maxvit_dense_speed.py`
- `fake.evaluation.speed.benchmark_forward`

因此新增 kernel 路径应尽量作为 runtime 封装层挂到模型上，而不是改动主压缩/评估框架。

## 设计原则

1. FlashInfer 依赖集中封装，主框架只感知 `replace_with_nvfp4_kernels(model, ...)`。
2. 权重离线预打包，forward 只量化 activation 并调用 FP4 GEMM。
3. 保留当前 fake-quant checkpoint 产物，不破坏已有 accuracy/speed CSV。
4. 所有 kernel 路径必须可 fallback 到原始 PyTorch Linear，方便定位数值和环境问题。
5. benchmark 输出必须标出 backend、cuda module、flashinfer version、替换模块数、fallback 模块数。

## 实现阶段

### Phase 0: 环境探测与最小算子验证

新增脚本：
- `scripts/check_flashinfer_nvfp4.py`
- `scripts/slurm/check_flashinfer_nvfp4.sh`

验证内容：
- import `torch` / `flashinfer`，打印版本、`torch.version.cuda`、设备名、compute capability。
- 检查 `flashinfer.nvfp4_quantize`、`flashinfer.gemm.mm_fp4` 可用。
- 在 RTX 5090 上跑一个最小矩阵：
  - A: `[M, K]` bf16/fp16
  - B: `[N, K]` bf16/fp16
  - `nvfp4_quantize(A, ..., sfLayout=layout_128x4, do_shuffle=False)`
  - `nvfp4_quantize(B, ..., sfLayout=layout_128x4, do_shuffle=...)`
  - `mm_fp4(A_fp4, B_fp4.T, A_sf, B_sf.T, alpha, out_dtype=...)`
- 与 `A @ B.T` 比较误差和耗时。

Slurm 脚本默认先用 `module load cuda/12.9`，同时保留 `CUDA_MODULE=${CUDA_MODULE:-cuda/12.9}` 方便回退到 12.8。

### Phase 1: FlashInfer kernel adapter

新增包：
- `fake/kernels/__init__.py`
- `fake/kernels/flashinfer_nvfp4.py`

核心对象：
- `FlashInferNVFP4Linear(nn.Module)`
  - 从一个 `nn.Linear` 构造。
  - 保存 bias。
  - 初始化时将权重按 FlashInfer `mm_fp4` 要求量化/打包为 fp4 权重和 block scale。
  - forward 时把输入展平为二维 `[M, K]`，对 activation 做 `nvfp4_quantize`，调用 `mm_fp4`，加 bias，再 reshape 回原始 batch 维度。
- `replace_linear_with_flashinfer_nvfp4(model, selector, config)`
  - 按 MaxViT 当前 selector 替换除 head 外的 `nn.Linear`。
  - 返回替换报告：替换数、跳过数、跳过原因。

配置建议：
- `block_size=16`
- `sfLayout=layout_128x4`
- `out_dtype=torch.bfloat16` 或跟随模型 dtype
- `backend="auto"` 初始默认；保留 `cutlass`、`cudnn`、`trtllm` 可选项
- `use_nvfp4=True`
- `activation_global_scale_mode="dynamic"` 初始使用每次 forward 动态 max；后续评估 per-token activation。

关键风险：
- `mm_fp4` 对 B 的转置/列主序/scale layout 有严格要求，需要在 Phase 0 小矩阵中先固定正确的数据布局。
- MaxViT Linear 输入可能是 3D/4D token layout，adapter forward 必须只折叠最后一维。
- K/N 维度不满足 block size 对齐时要跳过或 padding；第一版优先跳过并记录。

### Phase 2: MaxViT runtime 对接

新增入口：
- `fake/models/maxvit_nvfp4.py` 或在 `fake/models/maxvit.py` 增加轻量 helper：
  - `load_maxvit_nvfp4_runtime(..., kernel_backend="flashinfer")`

新增 benchmark：
- `scripts/bench_maxvit_nvfp4_speed.py`
- `scripts/slurm/bench_maxvit_nvfp4_speed.sh`

脚本行为：
- 加载 dense MaxViT。
- 调用 FlashInfer NVFP4 替换器。
- 用现有 `benchmark_forward` 做同样 batch/input/warmup/iters。
- 输出到 `artifacts/results/maxvit_${variant}_nvfp4/speed.csv`。
- CSV 增加字段：
  - `kernel_backend`
  - `flashinfer_version`
  - `cuda_module`
  - `nvfp4_block_size`
  - `nvfp4_backend`
  - `replaced_linear_count`
  - `skipped_linear_count`
  - `fallback_count`

为了对比公平，dense 和 nvfp4 使用同一批参数：
- same variant
- same batch size
- same input size
- same dtype policy
- same warmup/iters
- 同一 compute node 上连续跑，减少节点波动。

### Phase 3: 数值正确性与端到端对比

新增验证：
- `scripts/compare_maxvit_nvfp4_outputs.py`
  - 随机输入下比较 dense vs nvfp4 logits 的 max/mean/RMSE/cosine/top1 agreement。
  - 可选小样本 ImageNet accuracy smoke test。

新增一键 Slurm：
- `scripts/slurm/bench_maxvit_dense_vs_nvfp4.sh`
  - 同一 job 内先 dense 后 nvfp4。
  - 输出两个 CSV，并打印简短 summary。

验收标准：
- tiny variant 能完成端到端 forward benchmark。
- 替换的 Linear 数量和 skipped 数量清晰可见。
- nvfp4 路径确实调用 FlashInfer FP4 GEMM，不是 fake quant fallback。
- speed CSV 可以直接比较 dense vs nvfp4 的 latency/images_per_sec。

### Phase 4: 1x1 Conv2d 与进一步优化

在 Linear 路径稳定后，再处理 MaxViT 的 1x1 Conv2d：
- 对 `kernel_size == 1`、`groups == 1` 的 Conv2d，将输入 `[B, C, H, W]` 转为 `[B*H*W, C]`，调用同一 NVFP4 GEMM，再恢复 `[B, Cout, H, W]`。
- 对 MBConv 中 spatial layout 的转置成本单独测速，避免把 layout 开销误认为 GEMM 性能问题。

后续优化方向：
- 缓存/复用 activation scale buffer，减少 allocation。
- 尝试 `per_token_activation=True`。
- 对比 `backend=auto/cutlass/cudnn/trtllm`。
- 评估 CUDA Graph 捕获 benchmark，减少 launch jitter。
- 如果动态 activation 量化成本过高，增加校准得到静态/半静态 activation scale 的实验分支。

## 文件改动清单

计划新增：
- `fake/kernels/__init__.py`
- `fake/kernels/flashinfer_nvfp4.py`
- `scripts/check_flashinfer_nvfp4.py`
- `scripts/compare_maxvit_nvfp4_outputs.py`
- `scripts/bench_maxvit_nvfp4_speed.py`
- `scripts/slurm/check_flashinfer_nvfp4.sh`
- `scripts/slurm/bench_maxvit_nvfp4_speed.sh`
- `scripts/slurm/bench_maxvit_dense_vs_nvfp4.sh`

计划少量修改：
- `fake/models/maxvit.py`：增加可选 nvfp4 runtime loader，或导出 selector helper。
- `fake/evaluation/speed.py`：必要时增加 optional benchmark metadata，不改变现有调用。
- `artifacts/results/summary.md`：后续记录 dense vs nvfp4 结果。

## 环境注意

FlashInfer 如果安装或运行需要 CUDA 12.9，则相关 Slurm 脚本使用：

```bash
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
module load "${CUDA_MODULE}"
```

不改已有 dense benchmark 的 CUDA 12.8 默认，避免影响历史结果。若 FlashInfer 官方 wheel/JIT cache 与当前 PyTorch CUDA 版本冲突，优先在 `wja-cospaq` 中独立确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
flashinfer show-config
```

计算节点无网络，因此 FlashInfer 的 cubin/JIT cache 需要提前在登录节点准备好，或确保 Slurm job 不触发联网下载。

## 开发记录

实现完成后，在 `dev/impls/006_dense_nvfp4_flashinfer_impl.md` 追加记录。若后续继续优化这个方向，没有新 plan 时也追加到同一个 impl 文件。
