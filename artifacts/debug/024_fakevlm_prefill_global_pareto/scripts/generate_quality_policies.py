#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common_fakevlm_pareto import DEBUG_ROOT, METHODS, f, parse_methods, policy_counts, write_csv, write_json, write_policy
from common_fakevlm_pareto import module_family, module_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FakeVLM mixed policies for quality-model fitting.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--ratios", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--random-policies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-policies", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    methods = parse_methods(args.methods)
    rows = read_local_rows(args.output_root)
    modules = sorted({row["module_name"] for row in rows}, key=lambda name: int(next(r for r in rows if r["module_name"] == name)["module_index"]))
    by_module_method = {(row["module_name"], row["method"]): row for row in rows}
    policies: list[dict[str, Any]] = []

    dense_modules = [policy_module(by_module_method[(name, "dense_bf16")], "dense_bf16") for name in modules]
    maybe_add_policy(args, policies, "policy_000_dense", dense_modules, 0, "dense")

    index = 1
    ratios = [float(item) for item in args.ratios.split(",") if item.strip()]
    for method in methods:
        if method == "dense_bf16":
            continue
        ranked = sorted(modules, key=lambda name: f(by_module_method[(name, method)], "output_rel_mse"))
        for ratio in ratios:
            count = round(len(modules) * ratio)
            selected = set(stratified_take(ranked, count))
            policy_modules = [
                policy_module(by_module_method[(name, method if name in selected else "dense_bf16")], method if name in selected else "dense_bf16")
                for name in modules
            ]
            maybe_add_policy(args, policies, f"policy_{index:03d}_{method}_r{ratio:g}", policy_modules, index, f"{method}_ratio_{ratio:g}")
            index += 1
            if reached_limit(args, policies):
                break
        if reached_limit(args, policies):
            break

    non_dense = [method for method in methods if method != "dense_bf16"]
    for _ in range(args.random_policies):
        if reached_limit(args, policies):
            break
        policy_modules = []
        for name in modules:
            method = random.choice(methods)
            policy_modules.append(policy_module(by_module_method[(name, method)], method))
        maybe_add_policy(args, policies, f"policy_{index:03d}_random", policy_modules, index, "random_mixed")
        index += 1

    if len(non_dense) >= 2:
        for ratio in (0.25, 0.5, 0.75, 1.0):
            if reached_limit(args, policies):
                break
            policy_modules = []
            for module_idx, name in enumerate(modules):
                if module_idx / max(len(modules), 1) > ratio:
                    method = "dense_bf16"
                else:
                    method = non_dense[module_idx % len(non_dense)]
                policy_modules.append(policy_module(by_module_method[(name, method)], method))
            maybe_add_policy(args, policies, f"policy_{index:03d}_round_robin_r{ratio:g}", policy_modules, index, f"round_robin_{ratio:g}")
            index += 1

    write_csv(args.output_root / "stratified" / "quality_policies.csv", policies)
    write_json(
        args.output_root / "stratified" / "quality_policies_metadata.json",
        {"methods": methods, "module_count": len(modules), "policy_count": len(policies), "ratios": ratios, "random_policies": args.random_policies},
    )
    print(f"wrote {len(policies)} quality policies")


def read_local_rows(output_root: Path) -> list[dict[str, Any]]:
    from common_fakevlm_pareto import read_csv

    path = output_root / "sensitivity" / "module_method_local_errors.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def stratified_take(items: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def policy_module(row: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "name": row["module_name"],
        "module_name": row["module_name"],
        "module_index": int(f(row, "module_index")),
        "layer": int(f(row, "layer")),
        "module_type": row.get("module_type") or module_type(row["module_name"]),
        "module_family": row.get("module_family") or module_family(row["module_name"]),
        "n": int(f(row, "out_features")),
        "k": int(f(row, "in_features")),
        "selected_method": method,
        "backend": method,
    }


def write_one(output_root: Path, family: str, name: str, modules: list[dict[str, Any]], index: int, label: str) -> dict[str, Any]:
    path = output_root / "policies" / family / f"{name}.json"
    summary = {"policy_index": index, "label": label, "backend_counts": policy_counts(modules)}
    write_policy(path, family=family, modules=modules, summary=summary)
    return {"policy_index": index, "policy_name": name, "policy_json": str(path), "label": label, "backend_counts": summary["backend_counts"]}


def maybe_add_policy(args: argparse.Namespace, policies: list[dict[str, Any]], name: str, modules: list[dict[str, Any]], index: int, label: str) -> None:
    if reached_limit(args, policies):
        return
    policies.append(write_one(args.output_root, "stratified", name, modules, index, label))


def reached_limit(args: argparse.Namespace, policies: list[dict[str, Any]]) -> bool:
    return args.max_policies is not None and len(policies) >= args.max_policies


if __name__ == "__main__":
    main()
