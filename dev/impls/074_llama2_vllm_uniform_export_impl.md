## 2026-07-07 - Llama2 vLLM uniform export tooling
- 开发目的：为 vLLM 后端导出 Llama2-7B 的 uniform sparse BF16、dense NVFP4、sparse NVFP4 模型。
- 修改内容：新增计划记录，准备实现基于现有 prepared artifacts 的 fused Llama vLLM checkpoint 导出脚本。
- 影响文件：`dev/plans/074_llama2_vllm_uniform_export_plan.md`，`dev/impls/074_llama2_vllm_uniform_export_impl.md`，`scripts/export_llama2_vllm_uniform_compressed.py`。
- 后续注意：本轮不导出 mixed max-speed/P024；sparse 导出必须使用 `prune=False`，避免重新剪枝。

## 2026-07-07 - Export script and dry-run validation
- 开发目的：落地 uniform vLLM checkpoint 导出入口，并在无 GPU/轻量 Python 环境下验证默认路径和目标目录。
- 修改内容：新增 `export_llama2_vllm_uniform_compressed.py`，支持 `--methods`、`--prepared-root`、`--output-root`、`--force`、`--dry-run`；从 prepared metadata 推断原始 Llama2 HF 路径；按 fused Llama 命名导出 `qkv_proj`、`gate_up_proj`、`o_proj`、`down_proj`；dense NVFP4/sparse BF16/sparse NVFP4 分别写入 reference 约定的 quantization config。
- 影响文件：`scripts/export_llama2_vllm_uniform_compressed.py`，`dev/impls/074_llama2_vllm_uniform_export_impl.md`。
- 后续注意：实际导出需要 RTX 5090/SM120 GPU 和包含 `safetensors`、`cutlass_wrapper` 的运行环境；登录节点已完成 `py_compile` 和 `--dry-run`。

## 2026-07-07 - Full local GPU export and validation
- 开发目的：在本机 RTX 5090 上实际导出 uniform vLLM checkpoint，并验证 reference 格式。
- 修改内容：使用 `cospaq` 环境和 GPU 0 执行 `export_llama2_vllm_uniform_compressed.py --force`；生成 `uniform_sparse_bf16`、`uniform_dense_nvfp4`、`uniform_sparse_nvfp4` 三个目录；对三个目录运行 reference `validate_export_format.py` 均通过；使用 `vllm` 环境对 `uniform_dense_nvfp4` 做 eager generation smoke，模型可加载并生成短文本。
- 影响文件：`artifacts/exports/vllm/llama2_7b_018/uniform_sparse_bf16/`，`artifacts/exports/vllm/llama2_7b_018/uniform_dense_nvfp4/`，`artifacts/exports/vllm/llama2_7b_018/uniform_sparse_nvfp4/`，`dev/impls/074_llama2_vllm_uniform_export_impl.md`。
- 后续注意：默认 vLLM 编译路径会在 `nvfp4_mytest` 的运行时 `cutlass_wrapper` import 上触发 TorchDynamo unsupported；使用 `enforce_eager=True` 可完成 smoke。三个导出目录的 `model.safetensors` 大小分别约为 7.274 GiB、3.881 GiB、4.070 GiB。
