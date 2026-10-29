import json
import csv
import subprocess
import sys
from pathlib import Path


def test_merge_generation_shards_sorts_and_preserves_rows(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "workflows/paper/runtime/merge_generation_shards.py"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text(json.dumps({"question_id": 2, "answer": "c"}) + "\n")
    second.write_text("\n".join([
        json.dumps({"question_id": 1, "answer": "b"}),
        json.dumps({"question_id": 0, "answer": "a"}),
    ]) + "\n")
    subprocess.run([sys.executable, str(script), str(first), str(second),
                    "--expected-count", "3", "--output", str(output)], check=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question_id"] for row in rows] == [0, 1, 2]


def test_policy_nll_never_uses_runtime_prune() -> None:
    script = (Path(__file__).resolve().parents[1] /
              "workflows/paper/runtime/run_policy_nll.py").read_text()
    assert 'export.append("--prune")' not in script


def test_solver_workflow_accounts_for_quality_intercept(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    actions = tmp_path / "actions.csv"
    with actions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "module_name", "prefill_method", "decode_method", "quality_cost",
            "quality_intercept", "latency_ms", "supported",
        ])
        writer.writeheader()
        writer.writerows([
            {"module_name": "model.layers.0.q_proj", "prefill_method": "dense_bf16",
             "decode_method": "dense_bf16", "quality_cost": 0.0,
             "quality_intercept": -0.05, "latency_ms": 10.0, "supported": True},
            {"module_name": "model.layers.0.q_proj", "prefill_method": "dense_nvfp4",
             "decode_method": "dense_nvfp4", "quality_cost": 0.1,
             "quality_intercept": -0.05, "latency_ms": 5.0, "supported": True},
        ])
    output = tmp_path / "solved"
    subprocess.run([
        sys.executable, str(root / "workflows/paper/solve_from_actions.py"),
        "--actions", str(actions), "--budgets", "0.05", "--scenario", "prefill_only",
        "--output", str(output),
    ], check=True)
    row = next(csv.DictReader((output / "predicted_points.csv").open(newline="")))
    assert float(row["quality_intercept"]) == -0.05
    assert float(row["action_quality_cost"]) == 0.1
    assert float(row["predicted_delta_nll"]) == 0.05


def test_canonical_paper_tables_are_source_generated() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, str(root / "workflows/paper/build_tables.py"),
        "--root", str(root / "results/paper"), "--check",
    ], check=True)
