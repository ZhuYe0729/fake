#!/usr/bin/env python3
"""Generate deterministic random, neighborhood, and suspicious policies."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from config import (
    DEBUG,
    DECODE,
    DECODE_PARENTS,
    DECODE_STEPS,
    EXP_057,
    EXP_061,
    KERNEL,
    METHODS,
    NEIGHBORHOOD_RATES,
    PREFILL,
    PREFILL_PARENTS,
    RANDOM_RATIOS,
    SEED,
    SOURCE_038,
    SUSPICIOUS_COUNT,
    TYPES,
    parent_path,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_policy(path: Path) -> dict:
    return json.loads(path.read_text())


def modules_from(policy: dict) -> list[str]:
    return sorted(
        policy["method_map"],
        key=lambda name: (
            int(name.split(".")[2]),
            TYPES.index(name.rsplit(".", 1)[-1]),
        ),
    )


def write_policy(
    root: Path,
    scenario: str,
    family: str,
    policy_id: str,
    mapping: dict[str, dict[str, str]],
    metadata: dict,
) -> dict[str, object]:
    payload = {
        "policy_id": policy_id,
        "scenario": scenario,
        "policy_kind": f"search_audit_{family}",
        "default_prefill_method": "dense_bf16",
        "default_decode_method": "dense_bf16",
        "modules_to_not_convert": ["lm_head"],
        "method_map": mapping,
        "search_metadata": metadata,
    }
    path = root / "policies" / family / f"{policy_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    prefill = Counter(item["prefill_method"] for item in mapping.values())
    decode = Counter(item["decode_method"] for item in mapping.values())
    return {
        "scenario": scenario,
        "family": family,
        "policy_id": policy_id,
        "policy_json": str(path),
        "policy_sha256": hashlib.sha256(encoded).hexdigest(),
        "parent": metadata.get("parent", ""),
        "replacement_ratio": metadata.get("replacement_ratio", ""),
        "mutation_rate": metadata.get("mutation_rate", ""),
        "suspicious_module_count": metadata.get("suspicious_module_count", ""),
        **{f"prefill_count_{method}": prefill[method] for method in METHODS},
        **{f"decode_count_{method}": decode[method] for method in METHODS},
    }


def action_tables(scenario: str):
    if scenario == "prefill_only":
        rows = read_csv(SOURCE_038 / "action_support.csv")
        table = {
            (row["module_name"], row["kernel"]): float(row["latency_ms"])
            for row in rows if row["supported"] == "True"
        }
        legal = {
            name: [
                (method, method)
                for method in METHODS if (name, KERNEL[method]) in table
            ]
            for name in sorted({row["module_name"] for row in rows})
        }
        latency = lambda name, pair: table[name, KERNEL[pair[0]]]
        quality = prefill_quality()
        qcost = lambda name, pair: quality[name, pair[0]]
    else:
        rows = read_csv(EXP_057 / "speed/action_support.csv")
        table = {
            (row["phase"], row["module_name"], row["kernel"]): float(row["latency_ms"])
            for row in rows if row["supported"] == "True"
        }
        names = sorted({row["module_name"] for row in rows})
        legal = {}
        for name in names:
            legal[name] = [
                (pre, dec)
                for pre in METHODS for dec in METHODS
                if ("prefill", name, KERNEL[pre]) in table
                and ("decode", name, KERNEL[dec]) in table
            ]
        latency = lambda name, pair: (
            table["prefill", name, KERNEL[pair[0]]]
            + DECODE_STEPS * table["decode", name, KERNEL[pair[1]]]
        )
        quality = phase_quality()
        qcost = lambda name, pair: quality["prefill", name, pair[0]] + quality["decode", name, pair[1]]
    return legal, latency, qcost


def local_errors(phase: str) -> dict[tuple[int, str, str], float]:
    out = {(bucket, typ, "dense_bf16"): 0.0 for bucket in range(4) for typ in TYPES}
    for method in METHODS[1:]:
        for row in read_csv(EXP_057 / f"local_errors/{phase}_{method}.csv"):
            out[int(row["layer_bucket"]), row["fused_type"], method] = float(row["output_rel_mse"])
    return out


def prefill_quality() -> dict[tuple[str, str], float]:
    model = json.loads((EXP_061 / "reports/quality/model.json").read_text())
    errors = local_errors("prefill")
    out = {}
    for layer in range(32):
        bucket = layer // 8
        for group, typ in (
            ("self_attn", "qkv_proj"), ("self_attn", "o_proj"),
            ("mlp", "gate_up_proj"), ("mlp", "down_proj"),
        ):
            name = f"model.layers.{layer}.{group}.{typ}"
            for method_index, method in enumerate(METHODS):
                coefficient = max(
                    0.0,
                    float(model["global"])
                    + float(model["method"][method_index])
                    + float(model["bucket"][bucket])
                    + float(model["type"][TYPES.index(typ)]),
                ) / float(model["feature_scale"])
                out[name, method] = errors[bucket, typ, method] * coefficient
    return out


def phase_quality() -> dict[tuple[str, str, str], float]:
    model = json.loads((EXP_057 / "reports/quality_coverage_holdout/model.json").read_text())
    out = {}
    for phase_index, phase in enumerate(("prefill", "decode")):
        errors = local_errors(phase)
        for layer in range(32):
            bucket = layer // 8
            for group, typ in (
                ("self_attn", "qkv_proj"), ("self_attn", "o_proj"),
                ("mlp", "gate_up_proj"), ("mlp", "down_proj"),
            ):
                name = f"model.layers.{layer}.{group}.{typ}"
                for method_index, method in enumerate(METHODS):
                    coefficient = (
                        float(model["coefficients"][phase_index][method_index][bucket][TYPES.index(typ)])
                        / float(model["feature_scale"])
                    )
                    out[phase, name, method] = errors[bucket, typ, method] * coefficient
    return out


def leverage_modules(scenario: str, modules: list[str], legal, latency, qcost) -> list[str]:
    scored = []
    for name in modules:
        baseline = ("dense_bf16", "dense_bf16")
        alternatives = [pair for pair in legal[name] if pair != baseline]
        gains = [latency(name, baseline) - latency(name, pair) for pair in alternatives]
        qualities = [max(0.0, qcost(name, pair)) for pair in alternatives]
        positive = [value for value in qualities if value > 0]
        min_quality = min(positive, default=1e-9)
        score = max(gains, default=0.0) / max(min_quality, 1e-9) + 0.1 * max(qualities, default=0.0)
        scored.append((score, name))
    return [name for _, name in sorted(scored, reverse=True)[:SUSPICIOUS_COUNT]]


def map_from_pairs(pairs: dict[str, tuple[str, str]]) -> dict[str, dict[str, str]]:
    return {
        name: {"prefill_method": pair[0], "decode_method": pair[1]}
        for name, pair in pairs.items()
    }


def generate_scenario(scenario: str, root: Path, parents: tuple[str, ...]) -> list[dict[str, object]]:
    rng = random.Random(SEED + (0 if scenario == "prefill_only" else 10000))
    parent_payloads = {name: load_policy(parent_path(scenario, name)) for name in parents}
    modules = modules_from(next(iter(parent_payloads.values())))
    legal, latency, qcost = action_tables(scenario)
    baseline = ("dense_bf16", "dense_bf16")
    rows = []

    for index, ratio in enumerate(RANDOM_RATIOS, 1):
        count = max(1, round(len(modules) * ratio))
        selected = set(rng.sample(modules, count))
        pairs = {}
        for name in modules:
            choices = [pair for pair in legal[name] if pair != baseline]
            pairs[name] = rng.choice(choices) if name in selected else baseline
        rows.append(write_policy(
            root, scenario, "random", f"random_{index:03d}", map_from_pairs(pairs),
            {"seed": SEED, "replacement_ratio": ratio},
        ))

    neighborhood_index = 0
    for parent in parents:
        original = {
            name: (
                parent_payloads[parent]["method_map"][name]["prefill_method"],
                parent_payloads[parent]["method_map"][name]["decode_method"],
            )
            for name in modules
        }
        for rate in NEIGHBORHOOD_RATES:
            neighborhood_index += 1
            pairs = dict(original)
            for name in rng.sample(modules, max(1, round(len(modules) * rate))):
                pairs[name] = rng.choice([pair for pair in legal[name] if pair != pairs[name]])
            rows.append(write_policy(
                root, scenario, "neighborhood", f"neighborhood_{neighborhood_index:03d}",
                map_from_pairs(pairs), {"seed": SEED, "parent": parent, "mutation_rate": rate},
            ))

    suspicious = leverage_modules(scenario, modules, legal, latency, qcost)
    for index, parent in enumerate(parents, 1):
        pairs = {
            name: (
                parent_payloads[parent]["method_map"][name]["prefill_method"],
                parent_payloads[parent]["method_map"][name]["decode_method"],
            )
            for name in modules
        }
        for name in suspicious:
            pairs[name] = rng.choice([pair for pair in legal[name] if pair != pairs[name]])
        rows.append(write_policy(
            root, scenario, "suspicious", f"suspicious_{index:03d}", map_from_pairs(pairs),
            {"seed": SEED, "parent": parent, "suspicious_module_count": len(suspicious)},
        ))

    root.joinpath("search").mkdir(parents=True, exist_ok=True)
    with (root / "search/candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "search/metadata.json").write_text(json.dumps({
        "scenario": scenario,
        "seed": SEED,
        "families": {"random": 15, "neighborhood": 10, "suspicious": 5},
        "parents": list(parents),
        "suspicious_modules": suspicious,
    }, indent=2) + "\n")
    return rows


def main() -> None:
    prefill = generate_scenario("prefill_only", PREFILL, PREFILL_PARENTS)
    decode = generate_scenario("prefill_decode", DECODE, DECODE_PARENTS)
    print(json.dumps({"prefill_only": len(prefill), "prefill_decode": len(decode)}))


if __name__ == "__main__":
    main()
