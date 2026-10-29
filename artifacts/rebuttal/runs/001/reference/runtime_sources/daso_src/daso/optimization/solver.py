from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    prefill_method: str
    decode_method: str
    quality_cost: float
    latency_ms: float


@dataclass(frozen=True)
class ModuleChoices:
    name: str
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class Solution:
    quality_cost: float
    latency_ms: float
    actions: tuple[Action, ...]


def solve_budget(modules: list[ModuleChoices], budget: float, *, bins: int = 2000,
                 max_quality: float | None = None) -> Solution:
    if not modules or budget < 0 or bins <= 0:
        raise ValueError("invalid solver input")
    maximum = max_quality or sum(max(action.quality_cost for action in module.actions)
                                 for module in modules)
    scale = bins / max(maximum, 1e-12)
    # Round the budget down and positive action costs up.  This makes the
    # discretized problem conservative instead of silently relaxing the
    # caller's floating-point quality constraint.
    limit = min(bins, math.floor(budget * scale + 1e-12))
    states: dict[int, tuple[float, tuple[Action, ...], float]] = {0: (0.0, (), 0.0)}
    for module in modules:
        next_states: dict[int, tuple[float, tuple[Action, ...], float]] = {}
        for used, (latency, selected, exact_quality) in states.items():
            for action in module.actions:
                weight = 0 if action.quality_cost <= 0 else max(1, math.ceil(action.quality_cost * scale))
                total = used + weight
                if total > limit:
                    continue
                candidate = latency + action.latency_ms
                if total not in next_states or candidate < next_states[total][0]:
                    next_states[total] = (candidate, selected + (action,),
                                          exact_quality + action.quality_cost)
        states = next_states
        if not states:
            raise RuntimeError(f"no feasible policy after module {module.name}")
    _, (latency, actions, quality) = min(states.items(), key=lambda item: (item[1][0], item[0]))
    if quality > budget + 1e-12:
        raise RuntimeError(
            f"discretized solution violates exact quality budget: {quality} > {budget}")
    return Solution(quality, latency, actions)


def sweep_budgets(modules: list[ModuleChoices], budgets: list[float],
                  *, bins: int = 2000) -> list[Solution]:
    output: list[Solution] = []
    signatures: set[tuple[tuple[str, str], ...]] = set()
    maximum = sum(max(action.quality_cost for action in module.actions) for module in modules)
    for budget in budgets:
        result = solve_budget(modules, budget, bins=bins, max_quality=maximum)
        signature = tuple((a.prefill_method, a.decode_method) for a in result.actions)
        if signature not in signatures:
            signatures.add(signature)
            output.append(result)
    return output
