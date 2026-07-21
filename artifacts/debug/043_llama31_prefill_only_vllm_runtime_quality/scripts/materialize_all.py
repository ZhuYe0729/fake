#!/usr/bin/env python3
"""Materialize every missing published Llama3.1 prefill-only ours checkpoint."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEBUG = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((DEBUG / "manifest/policies.json").read_text())
    script = Path(__file__).with_name("materialize_ours.py")
    for item in manifest["policies"]:
        if item["kind"] != "ours":
            continue
        checkpoint = DEBUG.parents[2] / item["checkpoint"]
        if (checkpoint / "phase_hetero_policy.json").exists():
            print(f"reuse {item['label']}", flush=True)
            continue
        print(f"materialize {item['label']}", flush=True)
        subprocess.run([sys.executable, str(script), "--policy", item["label"]], check=True)


if __name__ == "__main__":
    main()
