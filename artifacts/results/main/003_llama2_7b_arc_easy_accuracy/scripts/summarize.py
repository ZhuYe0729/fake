#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import DEFAULT_MODEL_KEY, EXPERIMENT_ROOT, METHODS, model_result_root, write_csv, write_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Llama-2-7B arc_easy accuracy results.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = model_result_root(args.output_root, args.model)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        path = model_root / "methods" / method / "accuracy.json"
        if not path.exists():
            rows.append({"method": method, "status": "missing", "accuracy_json": str(path)})
            continue
        payload = json.loads(path.read_text())
        task = payload["results"].get("arc_easy", {})
        rows.append(
            {
                "method": method,
                "status": "ok",
                "acc": task.get("acc,none", task.get("acc")),
                "acc_norm": task.get("acc_norm,none", task.get("acc_norm")),
                "alias": task.get("alias", ""),
                "accuracy_json": str(path),
            }
        )
    summary_dir = model_root / "summary"
    write_csv(summary_dir / "accuracy_summary.csv", rows)
    write_json(summary_dir / "summary_metadata.json", {"timestamp": utc_now(), "methods": list(METHODS)})


if __name__ == "__main__":
    main()
