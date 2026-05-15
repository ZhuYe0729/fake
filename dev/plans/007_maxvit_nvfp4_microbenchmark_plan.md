# 007 MaxViT NVFP4 Microbenchmark Plan

## 目标

新增一个诊断型 benchmark，用于在 MaxViT tiny 上按典型输入尺寸拆解 FlashInfer NVFP4 路径的耗时：
- 所有被 NVFP4 替换的 `nn.Linear` 层逐层执行时间。
- FlashInfer `mm_fp4` GEMM 时间。
- FlashInfer `nvfp4_quantize` activation quant 时间。
- 其他辅助耗时，例如 activation global scale、scale+quant、alpha 计算、bias add、dense Linear baseline。

## 实现方案

1. 新增 `scripts/bench_maxvit_nvfp4_micro.py`。
   - 默认模型为 `maxvit tiny`。
   - 默认输入尺寸覆盖小/中/大：`3x224x224`、`3x448x448`、`3x672x672`。
   - 加载 FlashInfer NVFP4 MaxViT 后，注册 forward pre-hook，先用每个输入尺寸跑一次模型，捕获所有 `FlashInferNVFP4Linear` 的真实输入 shape。
   - 对每一层用捕获到的 shape 构造随机输入，分别测：
     - `layer_forward_ms`
     - `forward_like_2d_ms`
     - `activation_global_scale_ms`
     - `activation_quant_only_ms`
     - `activation_scale_plus_quant_ms`
     - `alpha_ms`
     - `gemm_only_ms`
     - `bias_add_ms`
     - `dense_linear_ms`
   - 输出逐层 CSV，并打印每个输入尺寸的简短汇总。

2. 新增 `scripts/README.md`。
   - 说明这些 MaxViT/NVFP4 benchmark 脚本的用途。
   - 给出计算节点/Slurm 中的典型用法。
   - 说明输出 CSV 的主要字段和解释口径。

3. 追加 `dev/impls/007_maxvit_nvfp4_microbenchmark_impl.md` 开发记录。

## 注意事项

- benchmark 需要 CUDA 和 FlashInfer，应提交到 GPU 计算节点运行。
- 脚本只读模型权重并写 CSV，不改模型产物。
- 逐层 benchmark 数量较多，默认 warmup/iters 保持中等；精确实验可手动增大。
