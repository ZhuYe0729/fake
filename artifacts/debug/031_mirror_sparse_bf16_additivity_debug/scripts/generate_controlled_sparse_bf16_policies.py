#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PARETO_ROOT = ROOT / "artifacts" / "debug" / "030_mirror_global_pareto"
OUT_ROOT = ROOT / "artifacts" / "debug" / "031_mirror_sparse_bf16_additivity_debug"
sys.path.insert(0, str(PARETO_ROOT / "scripts"))

from common_mirror_pareto import policy_counts, read_csv, write_csv, write_json, write_policy  # noqa: E402


COUNTS = (45, 78, 112, 157, 190)
SEEDS = (11, 23, 37)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled MIRROR sparse_bf16 debug policies.")
    parser.add_argument("--pareto-root", type=Path, default=PARETO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    costs = read_csv(args.pareto_root / "costs_keyfix_genimage" / "batch_16" / "module_method_candidates.csv")
    modules = dense_bf16_modules(costs)
    by_name = {row["module_name"]: row for row in modules}
    out_dir = args.output_root / "policies" / "controlled_sparse_bf16"
    rows = []
    index = 0

    def add(label: str, sparse_names: set[str], scenario: dict[str, Any]) -> None:
        nonlocal index
        payload = []
        for module in modules:
            method = "sparse_bf16" if module["module_name"] in sparse_names else "dense_bf16"
            payload.append({**module, "selected_method": method, "backend": method})
        path = out_dir / f"policy_{index:03d}_{label}.json"
        summary = {"policy_index": index, "label": label, "backend_counts": policy_counts(payload)}
        write_policy(path, family="controlled_sparse_bf16_additivity", modules=payload, summary=summary, scenario=scenario)
        rows.append(
            {
                "policy_index": index,
                "policy_name": path.stem,
                "policy_json": str(path),
                "label": label,
                "backend_counts": summary["backend_counts"],
                "sparse_bf16_count": len(sparse_names),
                "scenario": scenario,
            }
        )
        index += 1

    add("dense_bf16_baseline", set(), {"mode": "single_forward", "debug": "controlled_sparse_bf16_baseline"})
    add("uniform_sparse_bf16", set(by_name), {"mode": "single_forward", "debug": "controlled_sparse_bf16_uniform"})

    ordered_lowerr = sorted(modules, key=lambda row: (float(row.get("quality_cost", 0.0)), int(float(row["module_index"]))))
    ordered_speed = sorted(modules, key=lambda row: (-float(row.get("latency_gain_vs_dense_default", 0.0)), int(float(row["module_index"]))))
    for count in COUNTS:
        add(f"lowerr_count_{count}", {row["module_name"] for row in ordered_lowerr[:count]}, {"mode": "single_forward", "debug": "controlled_sparse_bf16_lowerr", "count": count})
        add(f"speed_count_{count}", {row["module_name"] for row in ordered_speed[:count]}, {"mode": "single_forward", "debug": "controlled_sparse_bf16_speed", "count": count})
        for seed in SEEDS:
            rng = random.Random(seed + count * 1000)
            add(f"random_count_{count}_seed_{seed}", {row["module_name"] for row in rng.sample(modules, count)}, {"mode": "single_forward", "debug": "controlled_sparse_bf16_random", "count": count, "seed": seed})

    for start in (0, 8, 16, 24):
        names = {row["module_name"] for row in modules if start <= int(float(row["layer"])) <= start + 7}
        add(f"layer_{start:02d}_{start+7:02d}", names, {"mode": "single_forward", "debug": "controlled_sparse_bf16_layer_bucket", "start_layer": start, "end_layer": start + 7})

    for family in ("attention", "mlp"):
        names = {row["module_name"] for row in modules if row["module_family"] == family}
        add(f"{family}_only", names, {"mode": "single_forward", "debug": "controlled_sparse_bf16_family", "family": family})

    for typ in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        names = {row["module_name"] for row in modules if row["module_type"] == typ}
        add(f"type_{typ}", names, {"mode": "single_forward", "debug": "controlled_sparse_bf16_type", "module_type": typ})

    csv_path = args.output_root / "controlled_sparse_bf16" / "quality_policies.csv"
    write_csv(csv_path, rows)
    write_json(csv_path.with_name("quality_policies_metadata.json"), {"policy_count": len(rows), "counts": COUNTS, "seeds": SEEDS})
    print(f"wrote {len(rows)} policies to {csv_path}")


def dense_bf16_modules(cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in cost_rows if row["method"] == "dense_bf16"]
    out = []
    for row in sorted(rows, key=lambda item: int(float(item["module_index"]))):
        out.append(
            {
                "name": row["module_name"],
                "module_name": row["module_name"],
                "module_index": int(float(row["module_index"])),
                "layer": int(float(row["layer"])),
                "module_type": row["module_type"],
                "module_family": row["module_family"],
                "n": int(float(row["out_features"])),
                "k": int(float(row["in_features"])),
                "selected_method": "dense_bf16",
                "backend": "dense_bf16",
                "quality_cost": float(row["quality_cost"]),
                "latency_cost": float(row["latency_cost"]),
            }
        )
    return out


if __name__ == "__main__":
    main()
