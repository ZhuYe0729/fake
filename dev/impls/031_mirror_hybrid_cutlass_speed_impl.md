## 2026-05-27 - MIRROR hybrid CUTLASS speed entry
- 开发目的：为 MIRROR-DINOv3-Huge 增加手动 sparse BF16 / sparse NVFP4 per-layer 混合速度实验，验证不同 batch 下是否能超过已有 best single runtime。
- 修改内容：新增 MIRROR hybrid loader，支持 4 个手动方案并适配 LoRA `q/k/v.base_layer`；新增 benchmark 脚本与 Slurm 脚本，输出 hybrid CSV。
- 影响文件：`fake/models/mirror_cutlass_hybrid.py`、`scripts/bench_mirror_cutlass_hybrid_speed.py`、`scripts/slurm/bench_mirror_cutlass_hybrid_speed.sh`、`dev/plans/031_mirror_hybrid_cutlass_speed_plan.md`。
- 验证：`python -m py_compile fake/models/mirror_cutlass_hybrid.py scripts/bench_mirror_cutlass_hybrid_speed.py` 通过；`bash -n scripts/slurm/bench_mirror_cutlass_hybrid_speed.sh` 通过；`PYTHONPATH=. python scripts/bench_mirror_cutlass_hybrid_speed.py --help` 通过。
- 后续注意：现有 MIRROR single-method batch sweep 中 Sparse BF16 已在各 batch 领先，hybrid 是否超过 best single 需要 GPU 节点实测确认。
