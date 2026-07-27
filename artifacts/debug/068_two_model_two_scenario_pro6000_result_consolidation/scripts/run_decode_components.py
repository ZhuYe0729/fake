#!/usr/bin/env python3
"""Export and benchmark the 14 selected Pro 6000 decode table policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BUNDLE = Path(__file__).resolve().parents[1]
MEASURE = BUNDLE / "measurements/decode_components"
CUTLASS = ROOT / "fake/kernels/cutlass/cutlass_wrapper"
VLLM = Path("/root/workspaces/cospaq/vllm-cospaq")
CONFIG = {
    "llama2_7b_chat": {
        "source": ROOT / "artifacts/debug/065_llama2_pro6000_prefill_decode",
        "model": Path("/root/data/models/Llama-2-7b-chat-hf"),
    },
    "llama31_8b_instruct": {
        "source": ROOT / "artifacts/debug/066_llama31_pro6000_prefill_decode",
        "model": Path("/root/data/models/Meta-Llama-3.1-8B-Instruct"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(VLLM / "vllm"), str(VLLM), str(CUTLASS), str(ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env


def selected() -> list[dict[str, str]]:
    path = BUNDLE / "data/selected_results.csv"
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["scenario"] == "prefill_decode"]


def source_policy(source: Path, label: str) -> Path:
    if label.startswith("uniform_p"):
        return source / "inputs/policies" / f"{label.removeprefix('uniform_')}.json"
    return source / "runs/experiment/pareto/policies" / f"{label}.json"


def snapshot_policy(model: str, source: Path, label: str) -> Path:
    src = source_policy(source, label)
    dst = MEASURE / "policies" / model / f"{label}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256(dst) != sha256(src):
        raise RuntimeError(f"frozen policy differs from source: {dst}")
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst


def complete(path: Path, policy_sha: str) -> bool:
    try:
        row = json.loads(path.read_text())
        return (row["policy_sha256"] == policy_sha
                and row["execution"] == "one_vllm_process_per_sample"
                and row["warmup_iters_per_phase"] == 1
                and row["measured_iters_per_phase"] == 5
                and len(row["ttft_measured_ms"]) == len(row["e2e_measured_ms"]) == 5
                and row["rtx5090_protocol_match"] is True)
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_one(row: dict[str, str], gpu: str) -> None:
    model = row["model"]
    label = row["source_label"]
    cfg = CONFIG[model]
    source = cfg["source"]
    policy = snapshot_policy(model, source, label)
    policy_sha = sha256(policy)
    output = MEASURE / "runs" / model / label
    if complete(output / "summary.json", policy_sha):
        print(f"[{model}/{label}] already complete", flush=True)
        return
    checkpoint = MEASURE / "temporary" / model / label
    canonical = source / "runs/experiment/canonical/prepared"
    canonical_bf16 = canonical / "sparse_bf16/model.pt"
    canonical_nvfp4 = canonical / "sparse_nvfp4/model.pt"
    env = runtime_env(gpu)
    verify = [
        sys.executable, str(BUNDLE / "scripts/verify_component_checkpoint.py"),
        "--policy", str(policy), "--checkpoint", str(checkpoint),
        "--canonical-bf16", str(canonical_bf16),
        "--canonical-nvfp4", str(canonical_nvfp4),
    ]
    if checkpoint.exists():
        audit = subprocess.run(verify, env=env, text=True, capture_output=True)
        if audit.returncode != 0:
            shutil.rmtree(checkpoint)
    if not checkpoint.exists():
        print(f"[{model}/{label}] exporting checkpoint", flush=True)
        subprocess.run([
            sys.executable, str(BUNDLE / "scripts/export_phase_hetero_model.py"),
            "--model-path", str(cfg["model"]), "--policy-json", str(policy),
            "--output-dir", str(checkpoint), "--cutlass-wrapper-path", str(CUTLASS),
            "--canonical-sparse-bf16-state", str(canonical_bf16),
            "--canonical-sparse-nvfp4-state", str(canonical_nvfp4),
        ], check=True, env=env)
    audit = subprocess.run(verify, check=True, env=env, text=True, capture_output=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_audit.json").write_text(audit.stdout)
    try:
        subprocess.run([
            sys.executable, str(BUNDLE / "scripts/benchmark_decode_components.py"),
            "--checkpoint", str(checkpoint), "--output-dir", str(output),
            "--policy-sha256", policy_sha, "--label", label, "--model", model,
            "--vllm-root", str(VLLM), "--cutlass-wrapper-path", str(CUTLASS),
            "--gpu-memory-utilization", "0.80",
        ], check=True, env=env)
    except Exception:
        print(f"[{model}/{label}] failed; checkpoint retained for resume", flush=True)
        raise
    shutil.rmtree(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", choices=tuple(CONFIG))
    parser.add_argument("--label")
    args = parser.parse_args()
    rows = selected()
    if args.model:
        rows = [row for row in rows if row["model"] == args.model]
    if args.label:
        rows = [row for row in rows if row["source_label"] == args.label]
    if not rows:
        raise RuntimeError("no selected policies matched")
    for row in rows:
        run_one(row, args.gpu)
    subprocess.run([sys.executable, str(BUNDLE / "scripts/summarize_decode_components.py")], check=True)


if __name__ == "__main__":
    main()
