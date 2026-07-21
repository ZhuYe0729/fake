#!/usr/bin/env python3
"""Serially fill missing formal Pareto NLL rows without retaining checkpoints."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/results"
L2_MODEL = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
L3_MODEL = Path("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")


def item(model: str, point: int) -> dict[str, str]:
    if model == "llama2":
        policy = ROOT / "artifacts/debug/034_llama2_7b_chat_wikitext_pareto_solver/prefill_decode/pareto/policies" / f"point_{point:03d}.json"
        return {"model": str(L2_MODEL), "tokenizer": str(L2_MODEL), "policy": str(policy),
                "samples": str(ROOT / "artifacts/debug/033_llama2_7b_chat_wikitext_phase_nll_proxy/samples/wikitext_2048_80.pt"),
                "output": str(OUT / "llama2_prefill_decode" / f"point_{point:03d}_runtime.json"),
                "label": f"llama2_point_{point:03d}_runtime"}
    policy = ROOT / "artifacts/debug/039_llama31_8b_instruct_prefill_decode_pareto/pareto/policies" / f"point_{point:03d}.json"
    return {"model": str(L3_MODEL), "tokenizer": str(L3_MODEL), "policy": str(policy),
            "samples": str(ROOT / "artifacts/debug/044_llama_prefill_decode_vllm_nll/samples/llama31_wikitext_2048_80.pt"),
            "output": str(OUT / "llama31_prefill_decode" / f"point_{point:03d}_runtime.json"),
            "label": f"llama31_point_{point:03d}_runtime"}


# point 000 is policy-identical to dense BF16. Existing stored checkpoints cover
# L3 002/004/006/008; L2 003 and 011(max-speed) are already measured.
QUEUE = [*(item("llama2", point) for point in (1, 2, 4, 5, 6, 7, 8, 9, 10)),
         *(item("llama3", point) for point in (1, 5, 7, 9))]


def main() -> None:
    state_path = HERE.parent / "run_state" / "formal_queue.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"started_at": datetime.now(timezone.utc).isoformat(), "queue": QUEUE, "completed": [], "skipped": []}
    for entry in QUEUE:
        output = Path(entry["output"])
        if output.exists():
            state["skipped"].append(entry["label"])
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue
        command = [sys.executable, str(HERE / "stream_phase_policy_nll.py"),
                   "--model-path", entry["model"], "--tokenizer", entry["tokenizer"],
                   "--policy-json", entry["policy"], "--samples", entry["samples"],
                   "--output", entry["output"], "--label", entry["label"], "--blocks", "32"]
        subprocess.run(command, check=True)
        state["completed"].append(entry["label"])
        state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
