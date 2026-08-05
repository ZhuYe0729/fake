#!/usr/bin/env python3
"""Resumable launcher for the four debug-069 validation stages."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import (
    COSPAQ_PYTHON,
    DECODE,
    EXP_057,
    EXP_061,
    PREFILL,
)


HERE = Path(__file__).resolve().parent
STAGES = {
    "prefill-speed": HERE / "run_prefill_speed.py",
    "prefill-arc": HERE / "run_prefill_arc.py",
    "decode-speed": HERE / "run_decode_speed.py",
    "decode-iwslt": HERE / "run_decode_iwslt.py",
}


def candidates(root: Path, families: set[str]) -> list[dict[str, str]]:
    with (root / "search/candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if not families or row["family"] in families]


def complete(stage: str, root: Path, policy_id: str) -> bool:
    if stage == "prefill-speed":
        return (root / f"speed/{policy_id}.json").exists()
    if stage == "prefill-arc":
        return (root / f"quality/{policy_id}/arc_challenge/full/result.json").exists()
    if stage == "decode-speed":
        path = root / f"speed/runs/{policy_id}/summary.csv"
        if not path.exists():
            return False
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle), {})
        return row.get("status") == "OK"
    return (root / f"quality/results/{policy_id}/IWSLT/metrics.json").exists()


def run_logged(stage: str, root: Path, policy_id: str, gpu: str, extra: list[str] | None = None) -> str:
    log = root / f"logs/{stage}/{policy_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(COSPAQ_PYTHON), str(STAGES[stage]), "--policy", policy_id, "--gpu", gpu,
        *(extra or []),
    ]
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return policy_id


def failure_path(stage: str, root: Path, policy_id: str) -> Path:
    return root / f"status/{stage}/{policy_id}.json"


def record_failure(stage: str, root: Path, policy_id: str, error: subprocess.CalledProcessError) -> None:
    path = failure_path(stage, root, policy_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "policy_id": policy_id,
        "status": "failed",
        "returncode": error.returncode,
        "log": str(root / f"logs/{stage}/{policy_id}.log"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")


def clear_failure(stage: str, root: Path, policy_id: str) -> None:
    failure_path(stage, root, policy_id).unlink(missing_ok=True)


def run_anchor(stage: str, root: Path, gpu: str, suffix: str) -> None:
    if stage == "prefill-speed":
        policy = EXP_061 / "pareto/policies/point_000.json"
    else:
        policy = EXP_057 / "pareto/policies/point_000.json"
    name = f"dense_anchor_{suffix}"
    if complete(stage, root, name):
        return
    run_logged(stage, root, name, gpu, ["--policy-json", str(policy), "--output-name", name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--families", default="")
    parser.add_argument("--max-policies", type=int)
    parser.add_argument("--policies", default="")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    root = PREFILL if args.stage.startswith("prefill") else DECODE
    families = {value for value in args.families.split(",") if value}
    rows = candidates(root, families)
    selected = {value for value in args.policies.split(",") if value}
    if selected:
        rows = [row for row in rows if row["policy_id"] in selected]
    if args.max_policies is not None:
        rows = rows[:args.max_policies]
    if args.stage == "prefill-arc":
        rows = [row for row in rows if complete("prefill-speed", root, row["policy_id"])]
    elif args.stage == "decode-iwslt":
        rows = [row for row in rows if complete("decode-speed", root, row["policy_id"])]
    todo = [
        row["policy_id"] for row in rows
        if not complete(args.stage, root, row["policy_id"])
        and (args.retry_failed or not failure_path(args.stage, root, row["policy_id"]).exists())
    ]
    gpus = [value for value in args.gpus.split(",") if value]
    if not gpus:
        raise ValueError("no GPUs")

    speed = args.stage.endswith("speed")
    if speed:
        run_anchor(args.stage, root, gpus[0], "before")
        for policy_id in todo:
            try:
                print(run_logged(args.stage, root, policy_id, gpus[0]), flush=True)
                clear_failure(args.stage, root, policy_id)
            except subprocess.CalledProcessError as error:
                record_failure(args.stage, root, policy_id, error)
                print(f"FAILED {policy_id}: returncode={error.returncode}", flush=True)
        run_anchor(args.stage, root, gpus[0], "after")
    else:
        def run(index: int, policy_id: str) -> str:
            try:
                result = run_logged(args.stage, root, policy_id, gpus[index % len(gpus)])
                clear_failure(args.stage, root, policy_id)
                return result
            except subprocess.CalledProcessError as error:
                record_failure(args.stage, root, policy_id, error)
                return f"FAILED {policy_id}: returncode={error.returncode}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run, index, policy_id) for index, policy_id in enumerate(todo)]
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
    print(json.dumps({"stage": args.stage, "selected": len(rows), "new": len(todo)}))


if __name__ == "__main__":
    main()
