#!/usr/bin/env python3
"""Reuse the 058 12-policy anchor design under the corrected timing protocol."""
from __future__ import annotations
import csv
import json
import shutil
from scenario import EXP, SOURCE, BATCH, INPUT_TOKENS, REPEATS, MAX_BATCHED_TOKENS


def main() -> None:
    source = SOURCE / "speed/calibration"
    rows = list(csv.DictReader((source / "design.csv").open()))
    output = EXP / "speed/calibration"; output.mkdir(parents=True, exist_ok=True)
    with (output / "design.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    old = json.loads((source / "metadata.json").read_text())
    metadata = {**old, "source_design": str(source / "design.csv"),
                "measurement": "one loaded phase-vLLM engine; explicit prefill phase; one warmup then five timed requests",
                "batch": BATCH, "input_tokens": INPUT_TOKENS, "max_num_seqs": BATCH,
                "max_num_batched_tokens": MAX_BATCHED_TOKENS, "repeats": REPEATS,
                "historical_058_speed_labels_used": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    quality = EXP / "reports/quality"; quality.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "reports/quality/model.json", quality / "model.json")


if __name__ == "__main__": main()
