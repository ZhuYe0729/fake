#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from daso.config import load_config
from daso.runtime import audit_backends


def git_revision(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    backend = audit_backends(config.backend.vllm_root, config.backend.cutlass_wrapper_root)
    errors = []
    if not (config.model.path / "config.json").is_file():
        errors.append(f"missing model config: {config.model.path}")
    if not backend.ok:
        errors.append("external vLLM/CUTLASS backend contract is incomplete")
    gpu: dict[str, object] = {"checked": not args.no_gpu}
    if not args.no_gpu:
        import torch
        gpu.update({"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                    "names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]})
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable")
    payload = {
        "ok": not errors, "errors": errors, "python": platform.python_version(),
        "config": str(args.config.resolve()), "model": asdict(config.model),
        "workload": asdict(config.workload), "backend": asdict(backend), "gpu": gpu,
        "revisions": {"vllm": git_revision(config.backend.vllm_root),
                      "cutlass_wrapper": git_revision(config.backend.cutlass_wrapper_root)},
    }
    payload = json.loads(json.dumps(payload, default=str))
    output = args.output or config.backend.run_root / "validation/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

