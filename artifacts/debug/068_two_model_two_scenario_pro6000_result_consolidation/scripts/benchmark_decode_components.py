#!/usr/bin/env python3
"""RTX-5090-compatible isolated-process TTFT/E2E benchmark for one checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BATCH = 8
INPUT_SEQ = 2048
OUTPUT_SEQ = 64
WARMUPS = 1
MEASURED = 5
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-root", type=Path, default=Path("/root/workspaces/cospaq/vllm-cospaq"))
    parser.add_argument("--cutlass-wrapper-path", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--single-phase", choices=("ttft", "main"))
    parser.add_argument("--single-output", type=Path)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    sys.path[:0] = [str(args.vllm_root / "vllm"), str(args.vllm_root), str(args.cutlass_wrapper_path)]
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PHASE_HETERO_TRACE"] = "0"
    for name in ("PHASE_HETERO_MYTEST", "NVFP4_MYTEST", "MARLIN_NVFP4_MYTEST", "SPARSE_BF16_MYTEST", "SPARSE_NVFP4_MYTEST"):
        os.environ[f"{name}_CUTLASS_WRAPPER_PATH"] = str(args.cutlass_wrapper_path)


def shutdown(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    callback = getattr(core, "shutdown", None)
    if callable(callback):
        callback()


def run_single(args: argparse.Namespace) -> None:
    configure(args)
    from vllm_compat import assert_chunked_prefill_disabled, force_v1_chunked_prefill_disabled
    force_v1_chunked_prefill_disabled()
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization import phase_hetero_mytest

    output_seq = 1 if args.single_phase == "ttft" else OUTPUT_SEQ
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + BATCH * 1000003 + INPUT_SEQ)
    token_ids = torch.randint(100, 30000, (BATCH, INPUT_SEQ), generator=generator, dtype=torch.int64)
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in token_ids]
    llm = LLM(
        model=str(args.checkpoint), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=INPUT_SEQ + OUTPUT_SEQ, max_num_seqs=BATCH,
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        max_num_batched_tokens=BATCH * INPUT_SEQ,
    )
    assert_chunked_prefill_disabled(llm)
    phase_hetero_mytest.enable_phase_hetero()
    sampling = SamplingParams(max_tokens=output_seq, min_tokens=output_seq,
                              temperature=0.0, ignore_eos=True, detokenize=False)
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated = sum(len(item.outputs[0].token_ids) for item in outputs)
        properties = torch.cuda.get_device_properties(0)
        payload = {
            "elapsed_ms": elapsed_ms,
            "phase": args.single_phase,
            "batch": BATCH,
            "input_seq": INPUT_SEQ,
            "output_seq": output_seq,
            "generated_tokens": generated,
            "timing_scope": "generate_only_after_loaded_llm",
            "execution": "one_vllm_process_per_sample",
            "process_id": os.getpid(),
            "cuda_device_name": properties.name,
            "cuda_device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "chunked_prefill_enabled": False,
            "prefix_caching_enabled": False,
            "max_num_batched_tokens": BATCH * INPUT_SEQ,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        args.single_output.parent.mkdir(parents=True, exist_ok=True)
        args.single_output.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        shutdown(llm)


def valid_sample(path: Path, phase: str) -> bool:
    try:
        row = json.loads(path.read_text())
        expected_output = 1 if phase == "ttft" else OUTPUT_SEQ
        return (float(row["elapsed_ms"]) > 0 and row["phase"] == phase
                and row["batch"] == BATCH and row["input_seq"] == INPUT_SEQ
                and row["output_seq"] == expected_output
                and row["execution"] == "one_vllm_process_per_sample"
                and row["chunked_prefill_enabled"] is False
                and row["prefix_caching_enabled"] is False
                and row["max_num_batched_tokens"] == BATCH * INPUT_SEQ)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def measure_phase(args: argparse.Namespace, phase: str) -> tuple[list[float], list[dict[str, Any]]]:
    rows = []
    values = []
    for index in range(WARMUPS + MEASURED):
        warmup = index < WARMUPS
        ordinal = index if warmup else index - WARMUPS
        name = "warmup" if warmup else f"measured_{ordinal}"
        path = args.output_dir / "raw" / phase / f"{name}.json"
        if not valid_sample(path, phase):
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--checkpoint", str(args.checkpoint), "--output-dir", str(args.output_dir),
                "--policy-sha256", args.policy_sha256, "--label", args.label,
                "--model", args.model, "--vllm-root", str(args.vllm_root),
                "--cutlass-wrapper-path", str(args.cutlass_wrapper_path),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--single-phase", phase, "--single-output", str(path),
            ]
            print(f"[{args.model}/{args.label}] {phase} {name}: start", flush=True)
            subprocess.run(command, check=True)
            print(f"[{args.model}/{args.label}] {phase} {name}: complete", flush=True)
        row = json.loads(path.read_text())
        row.update({"warmup": warmup, "iteration": index})
        rows.append(row)
        if not warmup:
            values.append(float(row["elapsed_ms"]))
    return values, rows


def main() -> None:
    args = parse_args()
    if args.single_phase:
        if args.single_output is None:
            raise ValueError("--single-output is required with --single-phase")
        run_single(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ttft_values, ttft_rows = measure_phase(args, "ttft")
    e2e_values, main_rows = measure_phase(args, "main")
    ttft = statistics.median(ttft_values)
    e2e = statistics.median(e2e_values)
    tpot = (e2e - ttft) / (OUTPUT_SEQ - 1)
    if tpot <= 0:
        raise RuntimeError(f"non-positive TPOT: e2e={e2e}, ttft={ttft}")
    uuids = sorted({row["cuda_device_uuid"] for row in ttft_rows + main_rows})
    summary = {
        "model": args.model, "label": args.label, "policy_sha256": args.policy_sha256,
        "batch": BATCH, "input_seq": INPUT_SEQ, "output_seq": OUTPUT_SEQ,
        "warmup_iters_per_phase": WARMUPS, "measured_iters_per_phase": MEASURED,
        "execution": "one_vllm_process_per_sample",
        "ttft_measured_ms": ttft_values, "e2e_measured_ms": e2e_values,
        "ttft_median_ms": ttft, "e2e_median_ms": e2e, "tpot_ms": tpot,
        "ttft_mean_ms": statistics.mean(ttft_values), "e2e_mean_ms": statistics.mean(e2e_values),
        "cuda_device_uuids": uuids,
        "tpot_formula": "(median_e2e_ms - median_ttft_ms) / 63",
        "rtx5090_protocol_match": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{args.model}/{args.label}] summary: TTFT={ttft:.3f} E2E={e2e:.3f} TPOT={tpot:.3f}", flush=True)


if __name__ == "__main__":
    main()
