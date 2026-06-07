from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from fake.kernels.offline_hybrid_policy import HybridPolicy, LayerPolicyDecision, load_policy_json
from fake.models.qwen3_5_kernels import (
    QwenHybridDenseNVFP4Linear,
    QwenManualHybridLinear,
    _build_manual_backend_module,
    _load_wrapper,
    _policy_backend_to_manual,
    _resolve_parent,
)


@dataclass(frozen=True)
class LlamaPredictorHybridReplacementReport:
    config: dict[str, Any]
    replaced_linear_count: int
    skipped_linear_count: int
    skipped: list[dict[str, str]]
    backend_counts: dict[str, int]

    @property
    def backend(self) -> str:
        return "llama_predictor_hybrid"


def replace_linear_with_llama_predictor_hybrid(
    model: nn.Module,
    *,
    policy_path: str | Path,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> LlamaPredictorHybridReplacementReport:
    from fake.compression.modules import select_compressible_modules

    policy = load_policy_json(policy_path)
    decode_m_threshold = int(policy.scenario.get("m_decode", policy.scenario.get("batch_size", 1)))
    skipped: list[dict[str, str]] = []
    backend_counts: dict[str, int] = {}
    replaced = 0
    wrapper = _load_wrapper()

    selected = select_compressible_modules(model, "llama")
    targets = [(info.name, info.kind) for info in selected]
    del selected
    for module_name, kind in targets:
        if kind != "linear":
            skipped.append({"name": module_name, "reason": f"unsupported_kind:{kind}"})
            continue
        decision = _policy_decision_for_module(policy, module_name)
        if decision is None:
            skipped.append({"name": module_name, "reason": "missing_policy"})
            continue
        if decision.selected_prefill_backend is None or decision.selected_decode_backend is None:
            skipped.append({"name": module_name, "reason": decision.reason or "unselected_policy"})
            continue

        parent, child_name = _resolve_parent(model, module_name)
        linear = getattr(parent, child_name)
        if not isinstance(linear, nn.Linear):
            skipped.append({"name": module_name, "reason": f"not_linear:{type(linear).__name__}"})
            continue
        if int(decision.n) != int(linear.out_features) or int(decision.k) != int(linear.in_features):
            skipped.append(
                {
                    "name": module_name,
                    "reason": (
                        f"policy_shape_mismatch:policy=({decision.n},{decision.k}),"
                        f"model=({linear.out_features},{linear.in_features})"
                    ),
                }
            )
            continue

        prefill_backend = _policy_backend_to_manual(decision.selected_prefill_backend)
        decode_backend = _policy_backend_to_manual(decision.selected_decode_backend)
        try:
            if _is_shared_nvfp4_policy(prefill_backend, decode_backend):
                canonical = wrapper.canonical_from_linear(linear, device=linear.weight.device)
                setattr(
                    parent,
                    child_name,
                    QwenHybridDenseNVFP4Linear(
                        canonical,
                        decode_activation_dtype=activation_dtype,
                        marlin_m_threshold=decode_m_threshold,
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                    ),
                )
                key = f"{prefill_backend}/{decode_backend}"
                backend_counts[key] = backend_counts.get(key, 0) + 1
            else:
                needed = tuple(dict.fromkeys((prefill_backend, decode_backend)))
                modules = {
                    backend: _build_manual_backend_module(linear, backend, activation_dtype=activation_dtype)
                    for backend in needed
                }
                setattr(
                    parent,
                    child_name,
                    QwenManualHybridLinear(
                        in_features=linear.in_features,
                        out_features=linear.out_features,
                        prefill_backend=prefill_backend,
                        decode_backend=decode_backend,
                        decode_m_threshold=decode_m_threshold,
                        modules=modules,
                    ),
                )
                for backend in needed:
                    backend_counts[backend] = backend_counts.get(backend, 0) + 1
            replaced += 1
        except Exception as exc:
            skipped.append({"name": module_name, "reason": f"{type(exc).__name__}:{exc}"})

    return LlamaPredictorHybridReplacementReport(
        config={
            "decode_activation_dtype": str(activation_dtype),
            "decode_m_threshold": decode_m_threshold,
            "policy": "offline_hybrid_policy",
            "policy_path": str(policy_path),
            "policy_format": policy.policy_format,
            "include_conversion_cost": policy.include_conversion_cost,
            "scenario": policy.scenario,
        },
        replaced_linear_count=replaced,
        skipped_linear_count=len(skipped),
        skipped=skipped,
        backend_counts=backend_counts,
    )


def _policy_decision_for_module(policy: HybridPolicy, module_name: str) -> LayerPolicyDecision | None:
    for decision in policy.modules:
        if module_name == decision.name or module_name.endswith(f".{decision.name}"):
            return decision
    return None


def _is_shared_nvfp4_policy(prefill_backend: str, decode_backend: str) -> bool:
    return {prefill_backend, decode_backend} == {"dense_nvfp4", "marlin_nvfp4"}
