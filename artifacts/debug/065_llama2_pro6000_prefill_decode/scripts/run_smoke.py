#!/usr/bin/env python3
"""Two-GPU, non-paper closure smoke for six representative policies."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import json

from common import PROTOCOL, RUN, VALIDATION, gpu_list, runtime_env, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=",".join(gpu_list()))
    args = parser.parse_args()
    labels = ("p00", "p01", "p02", "p03", "p04", "p71")
    jobs = list(labels)
    gpus = [value for value in args.gpus.split(",") if value]
    workers = {}
    state = {"diagnostic_only": True, "blocks": 2, "warmups": 1, "measured_runs": 2,
             "completed": [], "failed": []}
    while jobs or workers:
        for gpu in gpus:
            if gpu not in workers and jobs:
                label = jobs.pop(0)
                env = runtime_env(); env["CUDA_VISIBLE_DEVICES"] = gpu
                command = [sys.executable, str(Path(__file__).with_name("closure_policy.py")),
                           "--policy", str(RUN / f"policies/prefill_decode/{label}.json"),
                           "--label", label, "--gpu", gpu, "--blocks", "2", "--runs", "2",
                           "--output-root", str(RUN / "smoke")]
                workers[gpu] = (label, subprocess.Popen(command, env=env))
        time.sleep(2)
        for gpu, (label, process) in list(workers.items()):
            if process.poll() is None:
                continue
            del workers[gpu]
            if process.returncode == 0:
                state["completed"].append(label)
            else:
                state["failed"].append({"label": label, "gpu": gpu, "exit_code": process.returncode})
            write_json(RUN / "smoke/state.json", state)
    if state["failed"]:
        raise RuntimeError(f"smoke failed: {state['failed']}")
    audits = {}
    for label in labels:
        payload = json.loads((RUN / f"smoke/{label}/nll.json").read_text())
        trace = payload["runtime"]["phase_trace_events"]
        audits[label] = {"avg_nll": payload["avg_nll"], "token_count": payload["token_count"],
                         "capacity": payload["runtime"]["max_num_batched_tokens"],
                         "trace": trace,
                         "speed_median_ms": json.loads((RUN / f"smoke/{label}/speed/summary.json").read_text())["median_ms"]}
        if (payload["token_count"] != 128 or
                payload["runtime"]["max_num_batched_tokens"] != PROTOCOL["teacher_forcing_capacity"] or
                trace.get("enter_decode") != 1 or trace.get("apply_decode") != 8064):
            raise RuntimeError(f"invalid smoke audit: {label}")
    write_json(VALIDATION / "smoke.json", {"ok": True, "diagnostic_only": True,
                                            "policies": audits})


if __name__ == "__main__":
    main()
