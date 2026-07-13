# Llama2-7B vLLM Uniform Compressed Export Plan

## Summary
- 本轮只导出 uniform 模型，不导出 mixed max-speed 和 P024；mixed 等 vLLM mixed loader/格式明确后再单独处理。
- 输出目录固定为 `artifacts/exports/vllm/llama2_7b_018/`。
- 导出 3 个 Hugging Face 风格 vLLM checkpoint：
  - `uniform_sparse_bf16`
  - `uniform_dense_nvfp4`
  - `uniform_sparse_nvfp4`

## Key Changes
- 新增导出脚本 `scripts/export_llama2_vllm_uniform_compressed.py`，参考 `references/vllm/vllm_compressed_export_reference`。
- 默认复用现有 prepared artifacts：
  - `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/sparse_bf16/model.pt`
  - `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/dense_nvfp4/model.pt`
  - `artifacts/results/main/003_llama2_7b_arc_easy_accuracy/prepared/sparse_nvfp4/model.pt`
- 导出格式使用 reference 推荐的 fused Llama 结构：
  - `q_proj + k_proj + v_proj` -> `qkv_proj`
  - `gate_proj + up_proj` -> `gate_up_proj`
  - `o_proj`、`down_proj` 保持独立
- `config.json` 写入对应 `quantization_config`：
  - dense NVFP4: `quant_method = "nvfp4_mytest"`
  - sparse BF16: `quant_method = "sparse_bf16_mytest"`
  - sparse NVFP4: `quant_method = "sparse_nvfp4_mytest"`
- 非 Linear 权重和 tokenizer/config assets 从原始 Llama2 HF 模型目录复制；原始 `.safetensors/.bin/.pt/.pth` 权重文件不复制。
- sparse 导出 pack 时使用 `prune=False`，保持和 018 真实验证一样：prepared 权重已经压缩，不重新剪枝。

## Test Plan
- 静态检查：`python -m py_compile scripts/export_llama2_vllm_uniform_compressed.py`。
- dry-run/manifest 检查：确认每个 uniform 导出覆盖 32 层、每层 4 个 fused vLLM Linear base。
- GPU 导出：在 RTX 5090 计算节点、`wja-cospaq`、CUDA 12.8 环境下生成 3 个 `model.safetensors`。
- 格式验证：对 3 个输出目录运行 `references/vllm/vllm_compressed_export_reference/validate_export_format.py`。
- 可选 smoke：用 vLLM 加载每个导出目录生成短文本，`tensor_parallel_size=1`。

## Assumptions
- 本轮明确不导出 mixed max-speed 和 P024。
- 原始 Llama2 HF 模型目录优先从 prepared metadata 的 `model_path` 读取；若不存在，脚本要求显式传 `--model-path`。
- 输出目录已有内容时，脚本默认拒绝覆盖；显式传 `--force` 才删除并重建对应导出子目录。
