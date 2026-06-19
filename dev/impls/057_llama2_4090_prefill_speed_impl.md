## 2026-06-19 - Initial 4090 prefill benchmark scaffold
- 开发目的：为 4090 平台准备 Llama2-7B prefill-only 速度测试，覆盖 dense bf16、sparse bf16 和 Marlin W4A16。
- 修改内容：新增独立 debug 实验目录、benchmark 脚本、`gpu_4090` SLURM 启动脚本和使用说明。
- 影响文件：`dev/plans/057_llama2_4090_prefill_speed_plan.md`，`dev/impls/057_llama2_4090_prefill_speed_impl.md`，`artifacts/debug/025_llama2_4090_prefill_speed/`。
- 后续注意：当前机器不具备目标 GPU/超算环境；正式速度结果需要转移到超算后通过 `sbatch` 运行生成。
