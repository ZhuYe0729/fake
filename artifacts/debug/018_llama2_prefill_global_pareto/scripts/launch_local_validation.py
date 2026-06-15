#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common_pareto import DEBUG_ROOT, f, read_csv, write_csv, write_json


GPU_ORDER = (7, 6, 5, 4, 3, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch local one-GPU validation jobs over selected Pareto points.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--kind", choices=["e2e", "quality"], required=True)
    parser.add_argument("--gpus", default=",".join(str(gpu) for gpu in GPU_ORDER))
    parser.add_argument("--points", default="validation", help="'validation' or comma-separated point indices.")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--quality-subdir", default="quality_points")
    parser.add_argument("--quality-output-name", default="pareto_quality_validation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = selected_points(args)
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    script = Path(__file__).resolve().parent / ("validate_pareto_e2e.py" if args.kind == "e2e" else "validate_pareto_quality.py")
    logs = args.output_root / "logs" / args.kind
    logs.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, int, int, Path, Path]] = []
    remaining = [
        point
        for point in points
        if args.rerun_existing or not point_output_path(args.output_root, args.kind, point, quality_subdir=args.quality_subdir).exists()
    ]
    finished: list[dict[str, object]] = []
    while remaining or running:
        while remaining and len(running) < len(gpus):
            point = remaining.pop(0)
            gpu = next_free_gpu(gpus, running)
            out_log = logs / f"point_{point:03d}.out"
            err_log = logs / f"point_{point:03d}.err"
            cmd = [
                sys.executable,
                str(script),
                "--output-root",
                str(args.output_root),
                "--gpu",
                str(gpu),
                "--points",
                str(point),
                "--point-output-only",
            ] + split_extra_args(args.extra_args)
            if args.kind == "quality":
                cmd.extend(["--quality-subdir", args.quality_subdir, "--quality-output-name", args.quality_output_name])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with out_log.open("w") as stdout, err_log.open("w") as stderr:
                proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[4], env=env, stdout=stdout, stderr=stderr)
            running.append((proc, point, gpu, out_log, err_log))
        still_running = []
        for proc, point, gpu, out_log, err_log in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, point, gpu, out_log, err_log))
            else:
                finished.append({"point_index": point, "gpu": gpu, "returncode": code, "stdout": str(out_log), "stderr": str(err_log)})
        running = still_running
        write_json(args.output_root / "logs" / f"launch_{args.kind}_status.json", {"remaining": remaining, "running": running_summary(running), "finished": finished})
        if remaining or running:
            time.sleep(args.poll_seconds)
    failures = [row for row in finished if row["returncode"] != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} {args.kind} validation jobs failed; see logs")
    merge_point_outputs(args.output_root, args.kind, quality_subdir=args.quality_subdir, quality_output_name=args.quality_output_name)
    print(f"finished {len(finished)} {args.kind} validation jobs")


def selected_points(args: argparse.Namespace) -> list[int]:
    if args.points != "validation":
        return [int(item) for item in args.points.split(",") if item.strip()]
    selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
    return [int(f(row, "point_index")) for row in selected]


def split_extra_args(raw: str) -> list[str]:
    return [item for item in raw.split(" ") if item]


def running_summary(running: list[tuple[subprocess.Popen, int, int, Path, Path]]) -> list[dict[str, object]]:
    return [{"pid": proc.pid, "point_index": point, "gpu": gpu, "stdout": str(out_log), "stderr": str(err_log)} for proc, point, gpu, out_log, err_log in running]


def next_free_gpu(gpus: list[int], running: list[tuple[subprocess.Popen, int, int, Path, Path]]) -> int:
    used = {gpu for _proc, _point, gpu, _out_log, _err_log in running}
    for gpu in gpus:
        if gpu not in used:
            return gpu
    raise RuntimeError("no free GPU")


def point_output_path(output_root: Path, kind: str, point: int, *, quality_subdir: str = "quality_points") -> Path:
    subdir = "e2e_points" if kind == "e2e" else quality_subdir
    return output_root / "validation" / subdir / f"point_{point:03d}.csv"


def merge_point_outputs(
    output_root: Path,
    kind: str,
    *,
    quality_subdir: str = "quality_points",
    quality_output_name: str = "pareto_quality_validation.csv",
) -> None:
    subdir = "e2e_points" if kind == "e2e" else quality_subdir
    out_name = "pareto_e2e_validation.csv" if kind == "e2e" else quality_output_name
    rows = []
    for path in sorted((output_root / "validation" / subdir).glob("point_*.csv")):
        rows.extend(read_csv(path))
    rows.sort(key=lambda row: int(f(row, "point_index")))
    write_csv(output_root / "validation" / out_name, rows)
    write_json(output_root / "validation" / f"{out_name.removesuffix('.csv')}_metadata.json", {"rows": len(rows), "source": subdir})


if __name__ == "__main__":
    main()
