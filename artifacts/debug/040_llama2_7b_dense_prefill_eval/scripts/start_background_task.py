#!/usr/bin/env python3
"""Start one evaluation worker in its own session and record its PID."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("evaluate_task.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--proxy", action="store_true")
    args = parser.parse_args()

    task_dir = ROOT / "tasks" / args.task / "full"
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "stdout_stderr_background.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.proxy:
        env.update(
            {
                "http_proxy": "http://127.0.0.1:8848",
                "https_proxy": "http://127.0.0.1:8848",
                "all_proxy": "socks5://127.0.0.1:8848",
                "HF_HUB_DISABLE_XET": "1",
            }
        )
    command = [
        sys.executable,
        str(RUNNER),
        "--task",
        args.task,
        "--output",
        str(task_dir / "result.json"),
        "--batch-size",
        args.batch_size,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    record = {
        "pid": worker.pid,
        "task": args.task,
        "physical_gpu": args.gpu,
        "command": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "proxy_enabled": args.proxy,
        "log": str(log_path),
    }
    (task_dir / "background_attempt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
