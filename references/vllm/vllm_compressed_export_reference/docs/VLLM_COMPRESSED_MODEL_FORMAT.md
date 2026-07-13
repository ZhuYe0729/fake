# vLLM compressed model format used by this repo

本文档描述当前仓库自定义 vLLM 后端读取压缩 Llama checkpoint 时需要的输出格式。另一仓库可以使用自己的校准/压缩算法，但最终模型目录必须满足这里的文件结构、`config.json` 和 safetensors 张量命名。

## Model directory

导出目录保持 Hugging Face checkpoint 形态：

```text
exported-model/
  config.json
  tokenizer.json
  tokenizer.model
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
  model.safetensors
  <optional manifest>.json
```

导出时复制原模型的非权重资产，跳过原始 `.safetensors/.bin/.pt/.pth` 和旧 index 文件。当前参考脚本统一写单文件 `model.safetensors`；如果之后要分片，需要同步生成正确的 `model.safetensors.index.json`。

## Supported quant_method values

`config.json` 的 `quantization_config.quant_method` 决定 vLLM 使用哪个后端：

| Method | Meaning | Required tensor suffixes per Linear |
| --- | --- | --- |
| `nvfp4_mytest` | dense NVFP4 W4A4，activation online quant | `.weight`, `.weight_scale`, `.weight_global_scale` |
| `sparse_bf16_mytest` | 2:4 sparse BF16 | `.sparse_weight`, `.metadata` |
| `sparse_nvfp4_mytest` | sparse NVFP4 W4A4 | `.sparse_weight`, `.metadata`, `.weight_scale`, `.weight_global_scale` |
| `marlin_nvfp4_mytest` | W4A16 NVFP4 weight-only，本地 Marlin 路径 | `.packed_weight`, `.weight_scale`, `.weight_global_scale` |
| `compressed-tensors` | vLLM 内置 W4A16 FP4 compressed-tensors 路径 | `.weight_packed`, `.weight_scale`, global scale tensor |

所有自定义方法默认跳过 `lm_head`，所以 `lm_head.weight` 保持原始未压缩张量。

## Recommended fused Llama format

默认 vLLM Llama architecture 使用 fused projections：

| HF source weights | vLLM saved base name |
| --- | --- |
| `model.layers.{i}.self_attn.q_proj.weight` + `k_proj.weight` + `v_proj.weight` concat on dim 0 | `model.layers.{i}.self_attn.qkv_proj` |
| `model.layers.{i}.mlp.gate_proj.weight` + `up_proj.weight` concat on dim 0 | `model.layers.{i}.mlp.gate_up_proj` |
| `model.layers.{i}.self_attn.o_proj.weight` | `model.layers.{i}.self_attn.o_proj` |
| `model.layers.{i}.mlp.down_proj.weight` | `model.layers.{i}.mlp.down_proj` |

例如 dense NVFP4 fused Llama 每层应包含：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.qkv_proj.weight_scale
model.layers.0.self_attn.qkv_proj.weight_global_scale
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.o_proj.weight_scale
model.layers.0.self_attn.o_proj.weight_global_scale
model.layers.0.mlp.gate_up_proj.weight
model.layers.0.mlp.gate_up_proj.weight_scale
model.layers.0.mlp.gate_up_proj.weight_global_scale
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.down_proj.weight_scale
model.layers.0.mlp.down_proj.weight_global_scale
```

非 Linear 权重保持原名，例如 embeddings、layernorm、rotary 以外的普通参数：

```text
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

## Config examples

Dense NVFP4 W4A4:

```json
{
  "quantization_config": {
    "quant_method": "nvfp4_mytest",
    "group_size": 16,
    "modules_to_not_convert": ["lm_head"]
  },
  "torch_dtype": "bfloat16"
}
```

Sparse BF16:

```json
{
  "quantization_config": {
    "quant_method": "sparse_bf16_mytest",
    "backend": "cusparselt",
    "modules_to_not_convert": ["lm_head"]
  },
  "torch_dtype": "bfloat16"
}
```

Sparse NVFP4:

```json
{
  "quantization_config": {
    "quant_method": "sparse_nvfp4_mytest",
    "modules_to_not_convert": ["lm_head"]
  },
  "torch_dtype": "bfloat16"
}
```

Local W4A16 NVFP4:

```json
{
  "quantization_config": {
    "quant_method": "marlin_nvfp4_mytest",
    "modules_to_not_convert": ["lm_head"]
  },
  "torch_dtype": "bfloat16"
}
```

vLLM built-in W4A16 compressed-tensors:

```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "ignore": ["lm_head"],
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": {
          "num_bits": 4,
          "type": "float",
          "symmetric": true,
          "strategy": "tensor_group",
          "group_size": 16,
          "dynamic": false
        }
      }
    }
  },
  "torch_dtype": "bfloat16"
}
```

## compressed-tensors W4A16 naming caveat

For fused default Llama, the reference exporter writes:

```text
model.layers.{i}.self_attn.qkv_proj.weight_packed
model.layers.{i}.self_attn.qkv_proj.weight_scale
model.layers.{i}.self_attn.q_proj.weight_global_scale
model.layers.{i}.self_attn.k_proj.weight_global_scale
model.layers.{i}.self_attn.v_proj.weight_global_scale
```

and similarly:

```text
model.layers.{i}.mlp.gate_up_proj.weight_packed
model.layers.{i}.mlp.gate_up_proj.weight_scale
model.layers.{i}.mlp.gate_proj.weight_global_scale
model.layers.{i}.mlp.up_proj.weight_global_scale
```

This matches vLLM's stacked loader behavior for default Llama. For non-fused models, each independent Linear uses its own `{base}.weight_global_scale`.

## No-fuse format

No-fuse is an experimental architecture registered in this repo as:

```json
"architectures": ["LlamaNoFuseForCausalLM"]
```

It does not use `qkv_proj` or `gate_up_proj`. Per layer it expects independent base names:

```text
model.layers.{i}.self_attn.q_proj
model.layers.{i}.self_attn.k_proj
model.layers.{i}.self_attn.v_proj
model.layers.{i}.self_attn.o_proj
model.layers.{i}.mlp.gate_proj
model.layers.{i}.mlp.up_proj
model.layers.{i}.mlp.down_proj
```

Append the suffixes from the method table above. No-fuse is useful for debugging and per-projection experiments, but previous throughput tests showed it is usually slower than fused, especially for W4A4 NVFP4 paths.

## Shape constraints from current vLLM backends

- `nvfp4_mytest`: TP=1, `K % 32 == 0`, `N % 32 == 0`; saved `.weight` shape is `[N, K / 2]` uint8.
- `sparse_bf16_mytest`: TP=1, `K % 64 == 0`, `N % 8 == 0`; saved tensors are backend-packed sparse weight and metadata.
- `sparse_nvfp4_mytest`: TP=1, `K % 64 == 0`; saved tensors are backend-packed sparse weight, metadata and scales.
- `marlin_nvfp4_mytest`: TP=1, `K % 128 == 0`, `N % 64 == 0`; saved tensors are Marlin-packed weight and scales.

The exact packed tensor shapes are determined by `cutlass_wrapper`; the reference scripts call the same pack/quantize functions used during earlier integration tests.

## Recommended implementation approach in the other repo

1. Produce calibrated/compressed tensors per original HF Linear.
2. Decide fused or no-fuse. Prefer fused unless the algorithm needs independent q/k/v or gate/up modules.
3. If fused, concatenate original floating or compressed source in logical HF order before final packing:
   - q, k, v on dim 0
   - gate, up on dim 0
4. Save each vLLM Linear under the base names and suffixes above.
5. Preserve all non-converted tensors from the source checkpoint.
6. Run `validate_export_format.py` from this reference folder.
7. Copy the exported model directory back to this vLLM repo for generation smoke and lm-eval.
