#!/usr/bin/env python3
"""Materialize reproducible calibrated sparse states for phase export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = Path(os.environ.get("COSPAQ_EXPERIMENT_DIR", ROOT / "artifacts/debug/054_llama2_prefill_canonical_sparse_quality_recalibration/llama2_7b_chat"))
MODEL = Path(os.environ.get("COSPAQ_MODEL_PATH", "/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf"))
PREPARER = ROOT / "artifacts/exports/vllm/baselines/llama2-7b-chat/scripts/prepare_uniform_compressed.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(method: str, gpu: int) -> None:
    command = [sys.executable, str(PREPARER), "--methods", method,
               "--output-root", str(EXPERIMENT / "canonical"), "--gpu", str(gpu),
               "--model-path", str(MODEL)]
    if method == "sparse_nvfp4":
        command.append("--sparse-nvfp4-prequant-only")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    states = {
        "sparse_bf16": EXPERIMENT / "canonical/prepared/sparse_bf16/model.pt",
        "sparse_nvfp4": EXPERIMENT / "canonical/prepared/sparse_nvfp4/model.pt",
    }
    for method, state in states.items():
        if args.skip_existing and state.exists():
            continue
        run(method, args.gpu)
    manifest = {
        method: {"state": str(state), "sha256": sha256(state)}
        for method, state in states.items()
    }
    (EXPERIMENT / "canonical/canonical_manifest.json").write_text(
        json.dumps({"states": manifest,
                    "sparse_bf16": "SparseGPT 2:4, pack only",
                    "sparse_nvfp4": "SparseGPT pairwise 4:8, single final phase quantization"},
                   indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
