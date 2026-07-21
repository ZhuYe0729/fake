#!/usr/bin/env python3
"""Fail-fast preflight for the two-environment phase-vLLM workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from common import BUNDLE, REPO, env_path, write_json


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def check_python(path: Path, role: str, imports: list[str]) -> dict[str, object]:
    code = (
        "import importlib,json,sys;"
        f"names={imports!r};"
        "out={'executable':sys.executable,'version':sys.version.split()[0]};"
        "\nfor name in names:\n"
        " try:\n  mod=importlib.import_module(name);out[name]=getattr(mod,'__version__','import-ok')\n"
        " except Exception as exc:\n  out[name]='ERROR:'+repr(exc)\n"
        "print(json.dumps(out))"
    )
    result = run([str(path), "-c", code])
    result["role"] = role
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--output", type=Path, default=BUNDLE / "validation/preflight.json")
    args = parser.parse_args()

    model = env_path("COSPAQ_MODEL_PATH")
    vllm_root = env_path("COSPAQ_VLLM_ROOT")
    run_root = env_path("COSPAQ_RUN_ROOT")
    cospaq_python = env_path("COSPAQ_PYTHON")
    vllm_python = env_path("VLLM_PYTHON")
    cutlass = REPO / "fake/kernels/cutlass/cutlass_wrapper"
    errors: list[str] = []
    required = {
        "repo": REPO,
        "model_config": model / "config.json",
        "vllm_root": vllm_root,
        "vllm_source": vllm_root / "vllm",
        "phase_exporter": vllm_root / "artifacts/dev/012_phase_hetero_linear/export_phase_hetero_model.py",
        "cutlass_wrapper": cutlass,
        "cospaq_python": cospaq_python,
        "vllm_python": vllm_python,
    }
    existence = {key: {"path": str(path), "exists": path.exists()} for key, path in required.items()}
    for key, value in existence.items():
        if not value["exists"]:
            errors.append(f"missing {key}: {value['path']}")

    config: dict[str, object] = {}
    if (model / "config.json").is_file():
        config = json.loads((model / "config.json").read_text())
        if config.get("model_type") != "llama" or config.get("num_hidden_layers") != 32:
            errors.append(f"unexpected model architecture: {config.get('model_type')}, layers={config.get('num_hidden_layers')}")

    custom_runtime_hits = []
    if (vllm_root / "vllm").is_dir():
        for needle in ("phase_hetero_mytest", "nvfp4_mytest", "sparse_bf16_mytest",
                       "sparse_nvfp4_mytest", "marlin_nvfp4_mytest"):
            probe = run(["rg", "-l", needle, str(vllm_root / "vllm")])
            custom_runtime_hits.append({"needle": needle, "matches": probe["stdout"].splitlines(), "returncode": probe["returncode"]})
            if probe["returncode"] not in (0,):
                errors.append(f"patched vLLM source does not contain {needle}")

    python_checks = []
    if cospaq_python and cospaq_python.is_file():
        python_checks.append(check_python(cospaq_python, "modeling/export", ["torch", "transformers", "numpy", "pandas"]))
    if vllm_python and vllm_python.is_file():
        python_checks.append(check_python(vllm_python, "phase-vLLM runtime", ["torch", "vllm", "transformers"]))
    for item in python_checks:
        if item["returncode"] != 0 or "ERROR:" in str(item["stdout"]):
            errors.append(f"interpreter check failed for {item['role']}")

    disk = shutil.disk_usage(run_root.parent if run_root.parent.exists() else REPO)
    disk_info = {"path": str(run_root), "free_gib": round(disk.free / 2**30, 2), "total_gib": round(disk.total / 2**30, 2)}
    if disk.free < 80 * 2**30:
        errors.append("less than 80 GiB free; canonical states plus temporary checkpoints are unsafe")

    gpu = None
    lock_path = cutlass / "artifacts/torch_extensions/lock"
    extension_lock = {"path": str(lock_path), "exists": lock_path.exists()}
    if lock_path.exists():
        extension_lock["age_seconds"] = round(time.time() - lock_path.stat().st_mtime, 1)
        if extension_lock["age_seconds"] > 1800:
            errors.append("stale-looking CUTLASS extension lock (>30 min); verify no compiler owns it before removal")
    if not args.no_gpu:
        if shutil.which("nvidia-smi") is None:
            errors.append("nvidia-smi not found")
        else:
            gpu = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap", "--format=csv,noheader,nounits"])
            if gpu["returncode"] != 0:
                errors.append("nvidia-smi failed")

    report = {
        "ok": not errors,
        "errors": errors,
        "environment": {key: os.environ.get(key) for key in (
            "COSPAQ_REPO_ROOT", "COSPAQ_VLLM_ROOT", "COSPAQ_MODEL_PATH", "COSPAQ_RUN_ROOT",
            "COSPAQ_CANONICAL_DIR", "COSPAQ_EXT_CACHE_ROOT", "COSPAQ_GPUS", "VLLM_USE_V1",
            "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")},
        "paths": existence,
        "model": {key: config.get(key) for key in ("model_type", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "vocab_size")},
        "python": python_checks,
        "custom_runtime": custom_runtime_hits,
        "disk": disk_info,
        "extension_lock": extension_lock,
        "gpu": gpu,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
