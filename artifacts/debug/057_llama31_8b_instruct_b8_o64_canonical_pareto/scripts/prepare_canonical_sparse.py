#!/usr/bin/env python3
"""Create the two canonical calibrated sparse states used by every 057 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "artifacts/debug/057_llama31_8b_instruct_b8_o64_canonical_pareto/llama31_8b_instruct"
PREPARER = ROOT / "artifacts/exports/vllm/baselines/llama3.1-8b-instruct/scripts/prepare_uniform_compressed.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    states = {method: EXP / f"canonical/prepared/{method}/model.pt"
              for method in ("sparse_bf16", "sparse_nvfp4")}
    for method, state in states.items():
        if args.skip_existing and state.exists():
            continue
        command = [sys.executable, str(PREPARER), "--methods", method,
                   "--output-root", str(EXP / "canonical"), "--gpu", str(args.gpu)]
        if method == "sparse_nvfp4":
            command.append("--sparse-nvfp4-prequant-only")
        subprocess.run(command, check=True)
    payload = {method: {"state": str(state), "sha256": sha256(state)}
               for method, state in states.items()}
    out = EXP / "canonical/canonical_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"states": payload,
        "sparse_bf16": "SparseGPT 2:4, packed once by the phase exporter",
        "sparse_nvfp4": "SparseGPT pairwise 4:8; final NVFP4 packing occurs once in the phase exporter"}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
