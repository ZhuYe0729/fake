## 2026-05-26 - DINOv3 hybrid CUTLASS speed scaffolding
- 开发目的：按 030 计划增加 DINOv3-7B 手动混合 sparse NVFP4 / sparse BF16 速度实验，先覆盖 batch 16 和 batch 32。
- 修改内容：新增 hybrid loader，按 `b16_manual` / `b32_manual` 方案从原始 dense BF16 Linear 直接替换为目标 CUTLASS sparse backend；新增 speed benchmark 脚本和 Slurm 脚本；记录 hybrid scheme、两类模块数量、替换数量和 latency。
- 影响文件：`fake/models/dinov3_cutlass_hybrid.py`、`scripts/bench_dinov3_vit7b16_cutlass_hybrid_speed.py`、`scripts/slurm/bench_dinov3_vit7b16_cutlass_hybrid_speed.sh`、`dev/plans/030_dinov3_hybrid_cutlass_speed_plan.md`。
- 验证：`python3 -m py_compile fake/models/dinov3_cutlass_hybrid.py scripts/bench_dinov3_vit7b16_cutlass_hybrid_speed.py` 通过；`bash -n scripts/slurm/bench_dinov3_vit7b16_cutlass_hybrid_speed.sh` 通过；`PYTHONPATH=. python scripts/bench_dinov3_vit7b16_cutlass_hybrid_speed.py --help` 在 `wja-cospaq` 中通过。
- 后续注意：需要提交到 RTX 5090 计算节点验证；期望 batch 16 为 sparse NVFP4 240 个 / sparse BF16 40 个，batch 32 为 sparse NVFP4 80 个 / sparse BF16 200 个。
