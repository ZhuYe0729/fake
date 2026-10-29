#!/usr/bin/env python3
"""Solve policies from a fully audited action table produced by the predictors."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from daso.optimization import Action, ModuleChoices, solve_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--budgets", default="0,.01,.02,.04,.07,.1,.15,.22,.32,.45,.65,.9,1.2,1.6,2.0")
    parser.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[Action]] = defaultdict(list)
    intercepts: set[float] = set()
    with args.actions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("supported", "True").lower() != "true":
                continue
            if not row.get("quality_intercept"):
                raise ValueError("action table is missing required quality_intercept")
            intercepts.add(float(row["quality_intercept"]))
            grouped[row["module_name"]].append(Action(
                row["prefill_method"], row["decode_method"],
                float(row["quality_cost"]), float(row["latency_ms"])))
    if len(intercepts) != 1:
        raise ValueError("action table must contain one consistent quality_intercept")
    quality_intercept = intercepts.pop()
    modules = [ModuleChoices(name, tuple(actions)) for name, actions in sorted(grouped.items())]
    quality_budgets = [float(value) for value in args.budgets.split(",")]
    action_budgets = [value - quality_intercept for value in quality_budgets]
    if any(value < 0 for value in action_budgets):
        raise ValueError("quality budget is below the model intercept")
    maximum = sum(max(action.quality_cost for action in module.actions) for module in modules)
    solved = []
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for quality_budget, action_budget in zip(quality_budgets, action_budgets, strict=True):
        solution = solve_budget(modules, action_budget, max_quality=maximum)
        signature = tuple((action.prefill_method, action.decode_method)
                          for action in solution.actions)
        if signature not in signatures:
            signatures.add(signature)
            solved.append((quality_budget, solution))
    policy_dir = args.output / "policies"; policy_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (quality_budget, solution) in enumerate(solved):
        policy_id = f"point_{index:03d}"
        method_map = {module.name: {"prefill_method": action.prefill_method,
                                   "decode_method": action.decode_method}
                      for module, action in zip(modules, solution.actions, strict=True)}
        policy = {"policy_id": policy_id, "scenario": args.scenario,
                  "default_prefill_method": "dense_bf16", "default_decode_method": "dense_bf16",
                  "modules_to_not_convert": ["lm_head"], "method_map": method_map}
        (policy_dir / f"{policy_id}.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        total_quality = quality_intercept + solution.quality_cost
        rows.append({"policy_id": policy_id, "quality_budget": quality_budget,
                     "quality_intercept": quality_intercept,
                     "action_quality_cost": solution.quality_cost,
                     "predicted_delta_nll": total_quality,
                     "predicted_latency_ms": solution.latency_ms})
    with (args.output / "predicted_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
