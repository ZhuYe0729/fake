#!/usr/bin/env python3
"""Validate copied hashes, key metrics, and the paper-facing TeX fragment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    manifest = json.loads((EXP / "source_manifest.json").read_text())
    for row in manifest["files"]:
        source = ROOT / row["source"]
        destination = EXP / row["destination"]
        if sha256(source) != row["sha256"] or sha256(destination) != row["sha256"]:
            raise AssertionError(f"provenance mismatch: {row}")

    metrics = list(csv.DictReader((EXP / "data/key_metrics.csv").open()))
    lookup = {(row["category"], row["variant"]): float(row["value"])
              for row in metrics}
    close(lookup[("Kernel prediction", "Direct fit")], 0.1480497969)
    close(lookup[("Kernel prediction", "+ Roofline prior")], 0.1077631261)
    close(lookup[("Kernel prediction", "+ Shape correction")], 0.0800919053)
    close(lookup[("E2E composition", "Raw E2E aggregation")], 0.1817319529)
    close(lookup[("E2E composition", "+ Monotonic E2E calibration")], 0.0830196810)
    close(lookup[("Quality prediction", "Local aggregation")], 0.0571360548)
    close(lookup[("Quality prediction", "+ Phase awareness")], 0.0412536458)
    close(lookup[("Runtime E2E", "Synchronous phase-boundary copy")], 1625.8701030)
    close(lookup[("Runtime E2E", "Asynchronous overlapped prefetch")], 1521.7288192)
    close(lookup[("Runtime overlap", "Exposed copy stall")], 1.9193600)
    close(lookup[("Runtime overlap", "Hidden ratio")], 0.9937629608)

    tex = (EXP / "ablation.tex").read_text()
    forbidden = ("[X", "[Y", "[Z", "[R]", "PLACEHOLDER", "TODO")
    if any(token in tex for token in forbidden):
        raise AssertionError("ablation.tex contains a placeholder")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    if not figures or any(not (EXP / figure).exists() for figure in figures):
        raise AssertionError(f"missing TeX figure: {figures}")
    required = ("14.80\\%", "8.01\\%", "18.17\\%", "8.30\\%",
                "0.0571", "0.0413", "6.41\\%", "99.38\\%")
    if any(token not in tex for token in required):
        raise AssertionError("ablation.tex is missing a measured result")
    if list(EXP.rglob("model.safetensors")):
        raise AssertionError("portable consolidation unexpectedly contains a checkpoint")
    print(json.dumps({
        "status": "OK",
        "manifest_files": len(manifest["files"]),
        "key_metrics": len(metrics),
        "tex_figures": figures,
    }, indent=2))


if __name__ == "__main__":
    main()

