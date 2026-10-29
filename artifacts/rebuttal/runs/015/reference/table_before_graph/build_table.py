"""Rebuild the complete main table from the frozen run013 evidence."""
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
runpy.run_path(str(ROOT / "artifacts/rebuttal/runs/013/code/write_report.py"), run_name="__main__")

if Path(__file__).with_name("TPOT_DIAGNOSIS.md").exists():
    with Path(__file__).with_name("README.md").open("a") as handle:
        handle.write("\n## TPOT 运行时补充诊断\n\n"
                     "run014 已确认 W4A16 的 GPU 加速被 eager 执行开销掩盖，"
                     "并验证均匀基线的原生 CUDA Graph 路径。"
                     "本表继续保留 run013 的原始协议和数值；"
                     "Graph 基线与异构策略需采用一致运行时后才能更新排名。"
                     "见 [诊断结果与适用范围](TPOT_DIAGNOSIS.md)。\n")
