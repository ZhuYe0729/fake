## 2026-05-15 - DINOv3 CUTLASS NVFP4 inference path
- 开发目的：为 DINOv3 ViT-7B/16 接入独立 CUTLASS dense NVFP4 kernel 推理路径，保留现有 dense、checkpoint fake-quant 和 FlashInfer 路径。
- 修改内容：新增 CUTLASS NVFP4 adapter、DINOv3 CUTLASS loader、end-to-end speed/accuracy 脚本和对应 Slurm 入口；DINOv3 dense loader 增加可选 `torch_dtype`，CUTLASS 路径直接 bf16 加载以降低 7B 模型显存峰值；新增 011 plan 文件并补充脚本文档。
- 影响文件：`dev/plans/011_dinov3_cutlass_nvfp4_inference_plan.md`、`fake/kernels/cutlass_nvfp4.py`、`fake/models/dinov3.py`、`fake/models/dinov3_cutlass_nvfp4.py`、`scripts/bench_dinov3_vit7b16_cutlass_nvfp4_speed.py`、`scripts/eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.py`、`scripts/slurm/bench_dinov3_vit7b16_cutlass_nvfp4_speed.sh`、`scripts/slurm/eval_dinov3_vit7b16_cutlass_nvfp4_accuracy.sh`、`scripts/README.md`。
- 后续注意：CUTLASS extension build 与真实 forward 需要在 RTX 5090 GPU 节点运行；登录节点只做导入和语法检查。
