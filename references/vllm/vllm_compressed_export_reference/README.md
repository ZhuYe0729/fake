# vLLM compressed model export reference

这个目录可以整体复制到另一个校准/压缩项目中。目标是让另一边按这里的格式导出 checkpoint，然后把导出后的模型目录拿回本仓库，用当前自定义 vLLM backend 直接推理或跑 `lm-eval`。

## Contents

- `docs/VLLM_COMPRESSED_MODEL_FORMAT.md`: vLLM 侧期望的模型目录、`config.json`、张量命名、fused/no-fuse 规则。
- `scripts/quantize_nvfp4_mytest.py`: dense NVFP4 W4A4 fused Llama 导出参考。
- `scripts/export_sparse_common.py`: sparse BF16 / sparse NVFP4 fused Llama 共享导出参考。
- `scripts/quantize_sparse_bf16_mytest.py`: sparse BF16 wrapper。
- `scripts/quantize_sparse_nvfp4_mytest.py`: sparse NVFP4 wrapper。
- `scripts/export_w4a16_common.py`: W4A16 NVFP4 fused Llama 导出参考，含本地 `marlin_nvfp4_mytest` 和 vLLM `compressed-tensors` 变体。
- `scripts/export_nofuse_single_method.py`: no-fuse Llama 导出参考。
- `validate_export_format.py`: 检查导出目录是否具备当前 vLLM 需要的 config 和 safetensors 张量名。

## Typical workflow

1. 在另一个项目中完成校准/压缩，得到每个 Linear 的压缩权重和 metadata。
2. 按 `docs/VLLM_COMPRESSED_MODEL_FORMAT.md` 写出 Hugging Face 风格模型目录。
3. 运行检查：
   ```bash
   python validate_export_format.py --model-dir /path/to/exported-model
   ```
4. 把导出模型目录复制回本仓库，然后用 vLLM 或 lm-eval 加载：
   ```bash
   conda run -n vllm python - <<'PY'
   from vllm import LLM, SamplingParams
   llm = LLM(model="/path/to/exported-model", dtype="bfloat16",
             tensor_parallel_size=1, max_model_len=4096)
   out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16))
   print(out[0].outputs[0].text)
   PY
   ```

## Important defaults

- 当前自定义后端都只验证过 `tensor_parallel_size=1`。
- 当前本地 kernel 路径主要面向 RTX 5090 / SM120。
- 默认推荐导出 vLLM fused Llama 结构：`qkv_proj = cat(q,k,v)`，`gate_up_proj = cat(gate,up)`。
- 只有确实需要逐投影独立控制时，才用 no-fuse `LlamaNoFuseForCausalLM`。
