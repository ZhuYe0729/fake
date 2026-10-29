from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendAudit:
    vllm_root: str
    cutlass_wrapper_root: str
    exporter: str
    phase_runtime_hits: int
    predictor: str
    ok: bool


def audit_backends(vllm_root: str | Path, cutlass_wrapper_root: str | Path) -> BackendAudit:
    vllm = Path(vllm_root).resolve()
    cutlass = Path(cutlass_wrapper_root).resolve()
    exporter = vllm / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py"
    predictor = cutlass / "modeling/kernel_predictor.py"
    hits = 0
    source = vllm / "vllm"
    if source.is_dir():
        completed = subprocess.run(["rg", "-l", "phase_hetero_mytest", str(source)],
                                   text=True, capture_output=True)
        if completed.returncode in (0, 1):
            hits = len([line for line in completed.stdout.splitlines() if line])
    return BackendAudit(str(vllm), str(cutlass), str(exporter), hits,
                        str(predictor), exporter.is_file() and predictor.is_file() and hits > 0)


def write_audit(audit: BackendAudit, output: str | Path) -> None:
    Path(output).write_text(json.dumps(asdict(audit), indent=2) + "\n")


def run_external(command: list[str], *, cwd: str | Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)

