#!/usr/bin/env python3
"""Copy the verified Llama3 canonical states from 066 into the isolated run."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from common import RUN, sha256, write_json

SOURCE = Path(os.environ.get(
    "COSPAQ_CANONICAL_SOURCE",
    "/root/workspaces/cospaq/fake/artifacts/debug/066_llama31_pro6000_prefill_decode/runs/experiment/canonical",
)).resolve()
EXPECTED = {
    "sparse_bf16": "a4e5405ce5288f79f49d5ad10203d19d12edec477801a41b5329a751ed52b3b2",
    "sparse_nvfp4": "b697cbdfbf09df35f9b1f5f5264845e5d80021e943303fa7b8001d3087cbd549",
}


def main() -> None:
    target = RUN / "canonical"
    copied = {}
    for method, expected in EXPECTED.items():
        source_dir = SOURCE / "prepared" / method
        source_model = source_dir / "model.pt"
        if sha256(source_model) != expected:
            raise RuntimeError(f"source hash mismatch for {method}")
        target_dir = target / "prepared" / method
        target_model = target_dir / "model.pt"
        if not target_model.exists():
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "--reflink=auto", "-a", str(source_dir), str(target_dir)], check=True)
        actual = sha256(target_model)
        if actual != expected:
            raise RuntimeError(f"copied hash mismatch for {method}: {actual}")
        copied[method] = {"source": str(source_model), "source_sha256": expected,
                          "target": str(target_model), "target_sha256": actual}
    calibration_source = SOURCE / "calibration"
    calibration_target = target / "calibration"
    if calibration_source.is_dir() and not calibration_target.exists():
        shutil.copytree(calibration_source, calibration_target)
    write_json(target / "provenance.json", {
        "strategy": "copied_verified_066_canonical",
        "source_root": str(SOURCE),
        "states": copied,
        "sparse_nvfp4_final_quantization": "deferred_to_phase_exporter",
    })
    print(json.dumps(copied, indent=2))


if __name__ == "__main__":
    main()
