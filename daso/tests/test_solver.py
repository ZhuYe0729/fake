from daso.optimization import Action, ModuleChoices, solve_budget, sweep_budgets


def modules() -> list[ModuleChoices]:
    return [ModuleChoices(f"m{index}", (
        Action("dense_bf16", "dense_bf16", 0.0, 10.0),
        Action("dense_nvfp4", "dense_nvfp4", 0.1, 5.0),
    )) for index in range(2)]


def test_budget_is_respected() -> None:
    tight = solve_budget(modules(), 0.0)
    relaxed = solve_budget(modules(), 0.2)
    assert tight.latency_ms == 20
    assert relaxed.latency_ms == 10
    assert relaxed.quality_cost <= 0.2


def test_sweep_deduplicates() -> None:
    results = sweep_budgets(modules(), [0, 0.05, 0.1, 0.2])
    assert 2 <= len(results) <= 3
    assert all(a.latency_ms >= b.latency_ms for a, b in zip(results, results[1:]))


def test_discretization_never_relaxes_budget() -> None:
    choices = [ModuleChoices("m0", (
        Action("dense_bf16", "dense_bf16", 0.0, 10.0),
        Action("dense_nvfp4", "dense_nvfp4", 0.21, 1.0),
    ))]
    result = solve_budget(choices, 0.19, bins=10, max_quality=1.0)
    assert result.quality_cost <= 0.19
    assert result.actions[0].prefill_method == "dense_bf16"


def test_boundary_solution_uses_exact_budget() -> None:
    result = solve_budget(modules(), 0.2, bins=7)
    assert result.quality_cost <= 0.2
