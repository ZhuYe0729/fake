#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common_search_audit import DEBUG_ROOT, read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch FakeVLM search-audit validation across GPUs.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--keys", default="", help="Optional comma-separated policy keys.")
    parser.add_argument("--families", default="", help="Optional comma-separated policy families.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise RuntimeError("no GPUs specified")
    policies = read_csv(args.output_root / "search" / "search_policies.csv")
    if args.families:
        families = {item.strip() for item in args.families.split(",") if item.strip()}
        policies = [row for row in policies if row["family"] in families]
    if args.keys:
        wanted = {item.strip() for item in args.keys.split(",") if item.strip()}
        policies = [row for row in policies if row["key"] in wanted]
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    done = existing_done(args.output_root)
    queue = [row for row in policies if args.overwrite or row["key"] not in done]
    skipped_existing = len(policies) - len(queue)
    logs_dir = args.output_root / "logs" / "validation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "validate_search_policy.py"
    active: dict[str, dict[str, Any]] = {}
    completed = 0
    failed = 0
    started_keys: list[str] = []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            row = queue.pop(0)
            key = row["key"]
            log_path = logs_dir / f"{key}.log"
            cmd = [sys.executable, str(script), "--output-root", str(args.output_root), "--key", key, "--gpu", "0"]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.extra_args:
                cmd.extend(args.extra_args.split())
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_f = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=args.output_root.parents[2], env=env)
            active[gpu] = {"proc": proc, "key": key, "log": log_path, "log_f": log_f, "started": time.time()}
            started_keys.append(key)
            print(f"[launch] physical_gpu={gpu} visible_gpu=0 key={key} log={log_path}")
        time.sleep(args.poll_seconds)
        for gpu, item in list(active.items()):
            proc = item["proc"]
            ret = proc.poll()
            if ret is None:
                continue
            item["log_f"].close()
            elapsed = time.time() - item["started"]
            if ret == 0:
                completed += 1
                print(f"[done] physical_gpu={gpu} key={item['key']} elapsed={elapsed:.1f}s")
            else:
                failed += 1
                print(f"[fail] physical_gpu={gpu} key={item['key']} ret={ret} log={item['log']}")
            del active[gpu]
    write_json(
        args.output_root / "logs" / "launch_validation_status.json",
        {
            "requested": len(policies),
            "skipped_existing": skipped_existing,
            "started": len(started_keys),
            "completed": completed,
            "failed": failed,
            "keys": started_keys,
        },
    )
    if failed:
        raise SystemExit(f"{failed} validation jobs failed")
    print(f"completed validation jobs={completed}")


def existing_done(output_root: Path) -> set[str]:
    path = output_root / "validation" / "policies"
    if not path.exists():
        return set()
    return {item.stem for item in path.glob("*.json")}


if __name__ == "__main__":
    main()
