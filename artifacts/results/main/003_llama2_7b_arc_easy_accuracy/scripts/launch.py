#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import COMPRESSED_METHODS, DEFAULT_MODEL_KEY, EXPERIMENT_ROOT, METHODS, model_result_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 003 Llama accuracy experiment steps.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--phase", choices=["prepare", "eval", "all", "summarize"], default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[7, 6, 5, 4, 3, 2])
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-used-mb", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    free_gpus = query_free_gpus(args.gpus, max_used_mb=args.max_used_mb)
    if not free_gpus and args.phase != "summarize":
        raise SystemExit(f"No free GPUs found in requested set {args.gpus}")
    print(f"requested_gpus={args.gpus} free_gpus={free_gpus}", flush=True)
    if args.phase in {"prepare", "all"}:
        prepare_methods = [m for m in args.methods if m in COMPRESSED_METHODS]
        run_scheduled(
            phase="prepare",
            methods=prepare_methods,
            gpus=free_gpus,
            command_builder=lambda method, gpu: prepare_command(
                script_dir,
                args.output_root,
                args.model,
                method,
                gpu,
                args.calib_samples,
                args.seq_len,
                args.skip_existing,
            ),
            log_builder=lambda method: model_result_root(args.output_root, args.model) / "prepared" / method / "stdout.log",
        )
    if args.phase in {"eval", "all"}:
        run_scheduled(
            phase="eval",
            methods=list(args.methods),
            gpus=free_gpus,
            command_builder=lambda method, gpu: eval_command(script_dir, args.output_root, args.model, method, gpu, args.limit),
            log_builder=lambda method: model_result_root(args.output_root, args.model) / "methods" / method / "stdout.log",
        )
    if args.phase in {"summarize", "all"}:
        subprocess.check_call([sys.executable, str(script_dir / "summarize.py"), "--output-root", str(args.output_root), "--model", args.model])


def prepare_command(
    script_dir: Path,
    output_root: Path,
    model: str,
    method: str,
    gpu: int,
    calib_samples: int,
    seq_len: int,
    skip_existing: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_dir / "prepare.py"),
        "--method",
        method,
        "--model",
        model,
        "--gpu",
        str(gpu),
        "--output-root",
        str(output_root),
        "--calib-samples",
        str(calib_samples),
        "--seq-len",
        str(seq_len),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def eval_command(script_dir: Path, output_root: Path, model: str, method: str, gpu: int, limit: int | None) -> list[str]:
    cmd = [
        sys.executable,
        str(script_dir / "eval.py"),
        "--method",
        method,
        "--model",
        model,
        "--gpu",
        str(gpu),
        "--output-root",
        str(output_root),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def query_free_gpus(requested: list[int], *, max_used_mb: int) -> list[int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        print(f"warning: nvidia-smi query failed ({type(exc).__name__}:{exc}); using requested GPUs", file=sys.stderr)
        return requested
    usage: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            usage[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    free = []
    for gpu in requested:
        used_mb, util = usage.get(gpu, (0, 0))
        if used_mb <= max_used_mb and util <= 10:
            free.append(gpu)
        else:
            print(f"skip busy gpu={gpu} used_mb={used_mb} util={util}", flush=True)
    return free


def run_scheduled(
    *,
    phase: str,
    methods: list[str],
    gpus: list[int],
    command_builder,
    log_builder,
) -> None:
    if not methods:
        return
    pending = list(methods)
    while pending:
        wave = pending[: len(gpus)]
        pending = pending[len(gpus) :]
        jobs = []
        for method, gpu in zip(wave, gpus):
            jobs.append((command_builder(method, gpu), log_builder(method)))
        print(f"{phase} wave methods={wave}", flush=True)
        run_parallel(jobs)


def run_parallel(jobs: list[tuple[list[str], Path]]) -> None:
    if not jobs:
        return
    procs = []
    log_handles = []
    for cmd, log_path in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w")
        log_handles.append(log_f)
        print("launch:", " ".join(cmd), "log=", log_path, flush=True)
        env = os.environ.copy()
        if "--gpu" in cmd:
            gpu = cmd[cmd.index("--gpu") + 1]
            env["CUDA_VISIBLE_DEVICES"] = gpu
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT))
    failures = []
    for proc, (cmd, _log_path) in zip(procs, jobs):
        code = proc.wait()
        if code != 0:
            failures.append((code, cmd))
    for log_f in log_handles:
        log_f.close()
    if failures:
        for code, cmd in failures:
            print(f"failed code={code}: {' '.join(cmd)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
