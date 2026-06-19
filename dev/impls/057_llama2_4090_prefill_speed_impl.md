## 2026-06-19 - Initial 4090 prefill benchmark scaffold
- 开发目的：为 4090 平台准备 Llama2-7B prefill-only 速度测试，覆盖 dense bf16、sparse bf16 和 Marlin W4A16。
- 修改内容：新增独立 debug 实验目录、benchmark 脚本、`gpu_4090` SLURM 启动脚本和使用说明。
- 影响文件：`dev/plans/057_llama2_4090_prefill_speed_plan.md`，`dev/impls/057_llama2_4090_prefill_speed_impl.md`，`artifacts/debug/025_llama2_4090_prefill_speed/`。
- 后续注意：当前机器不具备目标 GPU/超算环境；正式速度结果需要转移到超算后通过 `sbatch` 运行生成。

## 2026-06-19 - Add decode and mixed scenarios
- 开发目的：在已有 4090 Llama2-7B 测试基础上增加 decode-heavy 和 prefill+decode 场景，保持 dense bf16、sparse bf16、Marlin W4A16 三种方法一致。
- 修改内容：benchmark 脚本新增多场景 preset、decode timing、E2E 汇总、per-scenario 输出目录和 combined summary；SLURM 脚本默认运行 `prefill_only decode_heavy prefill_decode`。
- 影响文件：`artifacts/debug/025_llama2_4090_prefill_speed/scripts/bench_llama2_4090_prefill_speed.py`，`artifacts/debug/025_llama2_4090_prefill_speed/run_llama2_4090_prefill_speed_4090.sh`，`artifacts/debug/025_llama2_4090_prefill_speed/README.md`。
- 后续注意：旧的 prefill-only 输出文件仍会被覆盖生成，新增主要结果看 `results/full_model_summary.csv` 和各场景子目录。
