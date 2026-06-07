from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_POLICY_KERNELS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
)
POLICY_FORMAT = "offline_hybrid_policy_v1"

_NVFP4_CONVERSIONS = {
    "dense_nvfp4": "canonical_to_cutlass",
    "marlin_nvfp4": "canonical_to_marlin",
}
_COMPATIBLE_PAIRS = {
    ("dense_nvfp4", "marlin_nvfp4"),
    ("marlin_nvfp4", "dense_nvfp4"),
}


@dataclass(frozen=True)
class LinearShapeSpec:
    name: str
    n: int
    k: int
    count: int = 1


@dataclass(frozen=True)
class ScenarioSpec:
    batch_size: int
    input_tokens: int
    output_tokens: int

    @property
    def m_prefill(self) -> int:
        return int(self.batch_size) * int(self.input_tokens)

    @property
    def m_decode(self) -> int:
        return int(self.batch_size)


@dataclass(frozen=True)
class KernelPredictionRecord:
    kernel: str
    supported: bool
    latency_ms: float | None
    reason: str = ""
    source: str = ""
    prediction_status: str = ""


@dataclass(frozen=True)
class StrategyCandidate:
    prefill_backend: str
    decode_backend: str
    prefill_latency_ms: float
    decode_latency_ms: float
    conversion_latency_ms: float
    total_latency_ms: float


@dataclass(frozen=True)
class LayerPolicyDecision:
    name: str
    n: int
    k: int
    count: int
    selected_prefill_backend: str | None
    selected_decode_backend: str | None
    selected_total_ms: float | None
    selected_prefill_ms: float | None
    selected_decode_ms: float | None
    selected_conversion_ms: float
    strategy_candidates: list[StrategyCandidate]
    prefill_candidates: list[KernelPredictionRecord]
    decode_candidates: list[KernelPredictionRecord]
    conversion_candidates: list[dict[str, Any]]
    reason: str = ""


@dataclass(frozen=True)
class HybridPolicy:
    policy_format: str
    scenario: dict[str, int]
    kernels: list[str]
    include_conversion_cost: bool
    modules: list[LayerPolicyDecision]


def select_offline_hybrid_policy(
    linears: Iterable[LinearShapeSpec],
    scenario: ScenarioSpec,
    predictor: Any,
    *,
    kernels: Sequence[str] | None = None,
    include_conversion_cost: bool = True,
) -> HybridPolicy:
    selected_kernels = list(DEFAULT_POLICY_KERNELS if kernels is None else kernels)
    modules = [
        _select_layer_policy(
            spec,
            scenario,
            predictor,
            selected_kernels,
            include_conversion_cost=include_conversion_cost,
        )
        for spec in linears
    ]
    return HybridPolicy(
        policy_format=POLICY_FORMAT,
        scenario={
            "batch_size": int(scenario.batch_size),
            "input_tokens": int(scenario.input_tokens),
            "output_tokens": int(scenario.output_tokens),
            "m_prefill": scenario.m_prefill,
            "m_decode": scenario.m_decode,
        },
        kernels=selected_kernels,
        include_conversion_cost=include_conversion_cost,
        modules=modules,
    )


def save_policy_json(policy: HybridPolicy, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy_to_dict(policy), indent=2) + "\n")


def load_policy_json(path: str | Path) -> HybridPolicy:
    payload = json.loads(Path(path).read_text())
    if payload.get("policy_format") != POLICY_FORMAT:
        raise ValueError(f"Unsupported offline hybrid policy format: {payload.get('policy_format')}")
    modules = [
        LayerPolicyDecision(
            name=str(row["name"]),
            n=int(row["n"]),
            k=int(row["k"]),
            count=int(row.get("count", 1)),
            selected_prefill_backend=row.get("selected_prefill_backend"),
            selected_decode_backend=row.get("selected_decode_backend"),
            selected_total_ms=_optional_float(row.get("selected_total_ms")),
            selected_prefill_ms=_optional_float(row.get("selected_prefill_ms")),
            selected_decode_ms=_optional_float(row.get("selected_decode_ms")),
            selected_conversion_ms=float(row.get("selected_conversion_ms", 0.0)),
            strategy_candidates=[
                StrategyCandidate(
                    prefill_backend=str(item["prefill_backend"]),
                    decode_backend=str(item["decode_backend"]),
                    prefill_latency_ms=float(item["prefill_latency_ms"]),
                    decode_latency_ms=float(item["decode_latency_ms"]),
                    conversion_latency_ms=float(item.get("conversion_latency_ms", 0.0)),
                    total_latency_ms=float(item["total_latency_ms"]),
                )
                for item in row.get("strategy_candidates", [])
            ],
            prefill_candidates=[
                KernelPredictionRecord(
                    kernel=str(item["kernel"]),
                    supported=bool(item["supported"]),
                    latency_ms=_optional_float(item.get("latency_ms")),
                    reason=str(item.get("reason", "")),
                    source=str(item.get("source", "")),
                    prediction_status=str(item.get("prediction_status", "")),
                )
                for item in row.get("prefill_candidates", [])
            ],
            decode_candidates=[
                KernelPredictionRecord(
                    kernel=str(item["kernel"]),
                    supported=bool(item["supported"]),
                    latency_ms=_optional_float(item.get("latency_ms")),
                    reason=str(item.get("reason", "")),
                    source=str(item.get("source", "")),
                    prediction_status=str(item.get("prediction_status", "")),
                )
                for item in row.get("decode_candidates", [])
            ],
            conversion_candidates=list(row.get("conversion_candidates", [])),
            reason=str(row.get("reason", "")),
        )
        for row in payload.get("modules", [])
    ]
    return HybridPolicy(
        policy_format=str(payload["policy_format"]),
        scenario={key: int(value) for key, value in payload.get("scenario", {}).items()},
        kernels=[str(kernel) for kernel in payload.get("kernels", DEFAULT_POLICY_KERNELS)],
        include_conversion_cost=bool(payload.get("include_conversion_cost", True)),
        modules=modules,
    )


def policy_to_dict(policy: HybridPolicy) -> dict[str, Any]:
    return asdict(policy)


def write_policy_csv(policy: HybridPolicy, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "n",
        "k",
        "count",
        "selected_prefill_backend",
        "selected_decode_backend",
        "selected_total_ms",
        "selected_prefill_ms",
        "selected_decode_ms",
        "selected_conversion_ms",
        "reason",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for module in policy.modules:
            writer.writerow({field: getattr(module, field) for field in fields})


def _select_layer_policy(
    spec: LinearShapeSpec,
    scenario: ScenarioSpec,
    predictor: Any,
    kernels: Sequence[str],
    *,
    include_conversion_cost: bool,
) -> LayerPolicyDecision:
    prefill = _predict_kernel_records(predictor.predict(scenario.m_prefill, int(spec.n), int(spec.k)), kernels)
    decode = _predict_kernel_records(predictor.predict(scenario.m_decode, int(spec.n), int(spec.k)), kernels)
    conversions = _predict_conversion_records(predictor, int(spec.n), int(spec.k), include_conversion_cost)
    conversion_costs = _conversion_cost_map(conversions)

    prefill_lat = _latency_by_kernel(prefill)
    decode_lat = _latency_by_kernel(decode)
    strategies: list[StrategyCandidate] = []
    for prefill_backend in kernels:
        for decode_backend in kernels:
            if not _is_legal_strategy(prefill_backend, decode_backend):
                continue
            if prefill_backend not in prefill_lat or decode_backend not in decode_lat:
                if int(scenario.output_tokens) != 0 or prefill_backend != decode_backend or prefill_backend not in prefill_lat:
                    continue
            conversion_ms = _strategy_conversion_cost(prefill_backend, decode_backend, conversion_costs)
            if conversion_ms is None:
                continue
            prefill_ms = prefill_lat[prefill_backend]
            decode_ms = 0.0 if int(scenario.output_tokens) == 0 else decode_lat[decode_backend]
            total_ms = int(spec.count) * (prefill_ms + int(scenario.output_tokens) * decode_ms + conversion_ms)
            strategies.append(
                StrategyCandidate(
                    prefill_backend=prefill_backend,
                    decode_backend=decode_backend,
                    prefill_latency_ms=prefill_ms,
                    decode_latency_ms=decode_ms,
                    conversion_latency_ms=conversion_ms,
                    total_latency_ms=total_ms,
                )
            )
    strategies.sort(key=lambda item: item.total_latency_ms)
    if not strategies:
        return LayerPolicyDecision(
            name=str(spec.name),
            n=int(spec.n),
            k=int(spec.k),
            count=int(spec.count),
            selected_prefill_backend=None,
            selected_decode_backend=None,
            selected_total_ms=None,
            selected_prefill_ms=None,
            selected_decode_ms=None,
            selected_conversion_ms=0.0,
            strategy_candidates=[],
            prefill_candidates=prefill,
            decode_candidates=decode,
            conversion_candidates=conversions,
            reason="no_legal_supported_strategy",
        )
    best = strategies[0]
    return LayerPolicyDecision(
        name=str(spec.name),
        n=int(spec.n),
        k=int(spec.k),
        count=int(spec.count),
        selected_prefill_backend=best.prefill_backend,
        selected_decode_backend=best.decode_backend,
        selected_total_ms=best.total_latency_ms,
        selected_prefill_ms=best.prefill_latency_ms,
        selected_decode_ms=best.decode_latency_ms,
        selected_conversion_ms=best.conversion_latency_ms,
        strategy_candidates=strategies,
        prefill_candidates=prefill,
        decode_candidates=decode,
        conversion_candidates=conversions,
    )


def _predict_kernel_records(selection: Any, kernels: Sequence[str]) -> list[KernelPredictionRecord]:
    by_kernel = {str(candidate.kernel): candidate for candidate in getattr(selection, "candidates", [])}
    rows = []
    for kernel in kernels:
        candidate = by_kernel.get(kernel)
        if candidate is None:
            rows.append(KernelPredictionRecord(kernel=kernel, supported=False, latency_ms=None, reason="missing_candidate"))
            continue
        rows.append(
            KernelPredictionRecord(
                kernel=kernel,
                supported=bool(candidate.supported),
                latency_ms=_optional_float(candidate.latency_ms),
                reason=str(getattr(candidate, "reason", "")),
                source=str(getattr(candidate, "source", "")),
                prediction_status=str(getattr(candidate, "prediction_status", "")),
            )
        )
    return rows


def _predict_conversion_records(
    predictor: Any,
    n: int,
    k: int,
    include_conversion_cost: bool,
) -> list[dict[str, Any]]:
    if not include_conversion_cost:
        return [
            {
                "conversion": conversion,
                "supported": True,
                "latency_ms": 0.0,
                "reason": "conversion_cost_disabled",
            }
            for conversion in sorted(set(_NVFP4_CONVERSIONS.values()))
        ]
    if not hasattr(predictor, "predict_conversion"):
        return [
            {
                "conversion": conversion,
                "supported": False,
                "latency_ms": None,
                "reason": "predictor_missing_predict_conversion",
            }
            for conversion in sorted(set(_NVFP4_CONVERSIONS.values()))
        ]
    return [asdict(candidate) for candidate in predictor.predict_conversion(n, k)]


def _latency_by_kernel(candidates: Sequence[KernelPredictionRecord]) -> dict[str, float]:
    return {
        candidate.kernel: float(candidate.latency_ms)
        for candidate in candidates
        if candidate.supported and candidate.latency_ms is not None
    }


def _conversion_cost_map(candidates: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        str(candidate["conversion"]): float(candidate["latency_ms"])
        for candidate in candidates
        if candidate.get("supported") and candidate.get("latency_ms") is not None
    }


def _is_legal_strategy(prefill_backend: str, decode_backend: str) -> bool:
    return prefill_backend == decode_backend or (prefill_backend, decode_backend) in _COMPATIBLE_PAIRS


def _strategy_conversion_cost(
    prefill_backend: str,
    decode_backend: str,
    conversion_costs: dict[str, float],
) -> float | None:
    needed = {
        _NVFP4_CONVERSIONS[backend]
        for backend in (prefill_backend, decode_backend)
        if backend in _NVFP4_CONVERSIONS
    }
    total = 0.0
    for conversion in needed:
        if conversion not in conversion_costs:
            return None
        total += conversion_costs[conversion]
    return total


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
