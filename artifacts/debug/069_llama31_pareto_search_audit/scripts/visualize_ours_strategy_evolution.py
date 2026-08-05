#!/usr/bin/env python3
"""Visualize layer-wise action changes along the measured ours frontiers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import DECODE, EXP_057, EXP_061, EXP_068, METHODS, PREFILL


MODULES = (
    ("qkv_proj", "QKV projection", "self_attn"),
    ("o_proj", "Attention output", "self_attn"),
    ("gate_up_proj", "MLP gate / up", "mlp"),
    ("down_proj", "MLP down", "mlp"),
)
METHOD_COLORS = {
    "dense_bf16": "#e5e7eb",
    "dense_nvfp4": "#3b82f6",
    "sparse_bf16": "#22c55e",
    "sparse_nvfp4": "#8b5cf6",
    "w4a16_ours": "#f59e0b",
}
METHOD_LABELS = {
    "dense_bf16": "Dense BF16",
    "dense_nvfp4": "Dense NVFP4",
    "sparse_bf16": "Sparse BF16",
    "sparse_nvfp4": "Sparse NVFP4",
    "w4a16_ours": "W4A16 (ours)",
}
METHOD_INDEX = {method: index for index, method in enumerate(METHODS)}
CMAP = ListedColormap([METHOD_COLORS[method] for method in METHODS])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def policy_path(scenario: str, policy_id: str) -> Path:
    if scenario == "prefill_only":
        candidates = (
            EXP_068 / f"policies/{policy_id}.json",
            EXP_061 / f"pareto/policies/{policy_id}.json",
        )
    else:
        candidates = (EXP_057 / f"pareto/policies/{policy_id}.json",)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"policy JSON not found for {scenario}/{policy_id}")


def module_name(layer: int, module_type: str, group: str) -> str:
    return f"model.layers.{layer}.{group}.{module_type}"


def load_frontier(scenario: str, root: Path) -> list[dict]:
    references = read_csv(root / "report/gap_to_reference.csv")
    items = []
    expected_modules = {
        module_name(layer, module_type, group)
        for layer in range(32)
        for module_type, _, group in MODULES
    }
    for row in references:
        path = policy_path(scenario, row["reference_policy"])
        payload = json.loads(path.read_text())
        mapping = payload["method_map"]
        if set(mapping) != expected_modules:
            raise ValueError(
                f"{path} has {len(mapping)} modules; expected the canonical 128"
            )
        for action in mapping.values():
            for phase in ("prefill", "decode"):
                method = action[f"{phase}_method"]
                if method not in METHOD_INDEX:
                    raise ValueError(f"unsupported method {method} in {path}")
        if scenario == "prefill_only" and any(
            action["prefill_method"] != action["decode_method"]
            for action in mapping.values()
        ):
            raise ValueError(f"phase mismatch in prefill-only policy {path}")
        items.append(
            {
                "policy_id": row["reference_policy"],
                "speedup": float(row["reference_speedup"]),
                "quality": float(row["reference_quality"]),
                "policy_json": str(path),
                "method_map": mapping,
            }
        )
    items.sort(key=lambda item: item["speedup"])
    if len({item["policy_id"] for item in items}) != len(items):
        raise ValueError(f"duplicate reference policy in {scenario}")
    return items


def short_policy_id(policy_id: str) -> str:
    if policy_id.startswith("point_"):
        return f"p{policy_id.removeprefix('point_')}"
    if policy_id.startswith("bridge_dense_nvfp4_"):
        return f"bridge_{policy_id.rsplit('_', 1)[-1]}"
    return policy_id


def point_labels(items: list[dict]) -> list[str]:
    return [
        f"{short_policy_id(item['policy_id'])}\n{item['speedup']:.3f}×"
        for item in items
    ]


def action_matrix(
    items: list[dict], phase: str, module_type: str, group: str
) -> np.ndarray:
    values = np.empty((32, len(items)), dtype=np.int8)
    for column, item in enumerate(items):
        for layer in range(32):
            name = module_name(layer, module_type, group)
            method = item["method_map"][name][f"{phase}_method"]
            values[layer, column] = METHOD_INDEX[method]
    return values


def draw_heatmap(ax, matrix: np.ndarray, labels: list[str]) -> None:
    ax.imshow(
        matrix,
        cmap=CMAP,
        vmin=-0.5,
        vmax=len(METHODS) - 0.5,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 32, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.28, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", labelsize=9)


def legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=METHOD_COLORS[method],
            edgecolor="#9ca3af",
            linewidth=0.7,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_prefill(items: list[dict]) -> None:
    labels = point_labels(items)
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 12.5), sharex=True)
    for axis, (module_type, title, group) in zip(axes, MODULES):
        draw_heatmap(
            axis,
            action_matrix(items, "prefill", module_type, group),
            labels,
        )
        axis.set_ylabel(f"{title}\nLayer", fontsize=11)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)
    axes[-1].tick_params(axis="x", labelrotation=52, labelsize=9)
    axes[-1].set_xlabel(
        "Ours frontier points ordered by measured E2E speedup  →",
        fontsize=13,
    )
    fig.suptitle(
        "Llama3.1-8B-Instruct ours policy evolution — prefill-only",
        fontsize=17,
        y=0.985,
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.012),
        frameon=True,
    )
    fig.tight_layout(rect=(0.02, 0.105, 0.995, 0.965), h_pad=0.7)
    save_figure(fig, PREFILL / "report/ours_strategy_evolution")


def plot_decode(items: list[dict]) -> None:
    labels = point_labels(items)
    phases = (("prefill", "Prefill phase"), ("decode", "Decode phase"))
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(15.0, 12.5),
        sharex="col",
        sharey="row",
    )
    for row, (module_type, module_title, group) in enumerate(MODULES):
        for column, (phase, phase_title) in enumerate(phases):
            axis = axes[row, column]
            draw_heatmap(
                axis,
                action_matrix(items, phase, module_type, group),
                labels,
            )
            if row == 0:
                axis.set_title(phase_title, fontsize=14)
            if column == 0:
                axis.set_ylabel(f"{module_title}\nLayer", fontsize=11)
        if row < len(MODULES) - 1:
            axes[row, 0].tick_params(axis="x", labelbottom=False)
            axes[row, 1].tick_params(axis="x", labelbottom=False)
    for axis in axes[-1]:
        axis.tick_params(axis="x", labelrotation=48, labelsize=9)
        axis.set_xlabel(
            "Ours frontier points ordered by measured E2E speedup  →",
            fontsize=12,
        )
    fig.suptitle(
        "Llama3.1-8B-Instruct ours policy evolution — prefill-decode",
        fontsize=17,
        y=0.985,
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.012),
        frameon=True,
    )
    fig.tight_layout(rect=(0.02, 0.105, 0.995, 0.96), h_pad=0.7, w_pad=0.8)
    save_figure(fig, DECODE / "report/ours_strategy_evolution")


def write_long_table(
    scenario: str, root: Path, items: list[dict], phases: tuple[str, ...]
) -> int:
    rows = []
    for item in items:
        for phase in phases:
            for module_type, _, group in MODULES:
                for layer in range(32):
                    name = module_name(layer, module_type, group)
                    rows.append(
                        {
                            "scenario": scenario,
                            "policy_id": item["policy_id"],
                            "speedup": item["speedup"],
                            "quality": item["quality"],
                            "phase": phase,
                            "layer": layer,
                            "module_type": module_type,
                            "method": item["method_map"][name][
                                f"{phase}_method"
                            ],
                            "policy_json": item["policy_json"],
                        }
                    )
    path = root / "report/ours_strategy_evolution.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    prefill = load_frontier("prefill_only", PREFILL)
    decode = load_frontier("prefill_decode", DECODE)
    plot_prefill(prefill)
    plot_decode(decode)
    prefill_rows = write_long_table(
        "prefill_only", PREFILL, prefill, ("prefill",)
    )
    decode_rows = write_long_table(
        "prefill_decode", DECODE, decode, ("prefill", "decode")
    )
    print(
        json.dumps(
            {
                "prefill_only": {
                    "policies": len(prefill),
                    "table_rows": prefill_rows,
                },
                "prefill_decode": {
                    "policies": len(decode),
                    "table_rows": decode_rows,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
