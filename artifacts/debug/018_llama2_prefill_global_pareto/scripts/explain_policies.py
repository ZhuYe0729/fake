#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_pareto import DEBUG_ROOT, METHODS, f, read_csv, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain Pareto policy method assignments and frontier transitions.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_dir = args.output_root / "pareto" / "policies"
    policies = []
    for path in sorted(policy_dir.glob("point_*.csv")):
        point = int(path.name.split("_")[1])
        rows = read_csv(path)
        policies.append((point, path, rows))
    family_rows = []
    type_rows = []
    layer_rows = []
    transition_rows = []
    previous: tuple[int, list[dict[str, Any]]] | None = None
    for point, path, rows in policies:
        family_rows.extend(group_counts(point, rows, "module_family"))
        type_rows.extend(group_counts(point, rows, "module_type"))
        layer_rows.extend(group_counts(point, rows, "layer"))
        if previous is not None:
            prev_point, prev_rows = previous
            transition_rows.extend(transitions(prev_point, point, prev_rows, rows))
        previous = (point, rows)
    write_csv(args.output_root / "summary" / "policy_family_method_counts.csv", family_rows)
    write_csv(args.output_root / "summary" / "policy_type_method_counts.csv", type_rows)
    write_csv(args.output_root / "summary" / "policy_layer_method_counts.csv", layer_rows)
    write_csv(args.output_root / "summary" / "policy_transition_summary.csv", transition_rows)
    write_json(
        args.output_root / "summary" / "policy_explanation_metadata.json",
        {
            "policies": len(policies),
            "family_rows": len(family_rows),
            "type_rows": len(type_rows),
            "layer_rows": len(layer_rows),
            "transition_rows": len(transition_rows),
        },
    )
    append_analysis(args.output_root / "summary" / "analysis.md", transition_rows)
    print(f"wrote explanations for {len(policies)} policies")


def group_counts(point: int, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        group = str(row.get(key, ""))
        method = str(row["selected_method"])
        item = groups.setdefault(
            (group, method),
            {
                "point_index": point,
                key: group,
                "method": method,
                "modules": 0,
                "latency_cost": 0.0,
                "quality_cost": 0.0,
            },
        )
        item["modules"] += 1
        item["latency_cost"] += f(row, "latency_cost")
        item["quality_cost"] += f(row, "quality_cost")
    return list(groups.values())


def transitions(prev_point: int, point: int, prev_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prev = {row["module_name"]: row for row in prev_rows}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        old = prev[row["module_name"]]
        old_method = old["selected_method"]
        new_method = row["selected_method"]
        if old_method == new_method:
            continue
        key = (old_method, new_method, row["module_family"])
        item = out.setdefault(
            key,
            {
                "from_point": prev_point,
                "to_point": point,
                "from_method": old_method,
                "to_method": new_method,
                "module_family": row["module_family"],
                "modules": 0,
                "latency_delta_ms": 0.0,
                "quality_delta": 0.0,
            },
        )
        item["modules"] += 1
        item["latency_delta_ms"] += f(row, "latency_cost") - f(old, "latency_cost")
        item["quality_delta"] += f(row, "quality_cost") - f(old, "quality_cost")
    return list(out.values())


def append_analysis(path: Path, transitions_rows: list[dict[str, Any]]) -> None:
    lines = path.read_text().rstrip().splitlines() if path.exists() else ["# Llama2 Prefill-Only Pareto Analysis"]
    lines.extend(["", "## Policy Explanation", ""])
    if not transitions_rows:
        lines.append("- No policy transitions found.")
    else:
        largest = sorted(transitions_rows, key=lambda row: abs(float(row["latency_delta_ms"])), reverse=True)[:8]
        for row in largest:
            lines.append(
                "- point "
                f"{row['from_point']}->{row['to_point']}: "
                f"{row['modules']} {row['module_family']} modules "
                f"{row['from_method']}->{row['to_method']}, "
                f"latency_delta_ms={float(row['latency_delta_ms']):.6f}, "
                f"quality_delta={float(row['quality_delta']):.6f}"
            )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
