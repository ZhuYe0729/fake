#!/usr/bin/env python3
"""Fail-fast environment and dependency audit for the isolated 066 run."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from common import (BERTSCORE_MODEL, BUNDLE, COSPAQ_PYTHON, CUTLASS,
                    EXPORTER, IWSLT_FILTER_TOKENIZER, MODEL, PMPD_DATA_ROOT, REPO,
                    VALIDATION, VLLM_PYTHON, VLLM_ROOT, command_output, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    required = {
        "repo": REPO, "model_config": MODEL / "config.json", "vllm_root": VLLM_ROOT,
        "vllm_source": VLLM_ROOT / "vllm/vllm", "exporter": EXPORTER,
        "cutlass": CUTLASS, "cospaq_python": COSPAQ_PYTHON, "vllm_python": VLLM_PYTHON,
        "wikitext_arrow": Path(os.environ.get("COSPAQ_WIKITEXT_ARROW", "")),
        "tokenizer_config": MODEL / "tokenizer_config.json",
        "policy_inputs": BUNDLE / "inputs/policies",
        "pmpd_data_root": PMPD_DATA_ROOT, "bertscore_model": BERTSCORE_MODEL,
        "iwslt_filter_tokenizer": IWSLT_FILTER_TOKENIZER,
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"missing {name}: {path}")
    model_config = json.loads((MODEL / "config.json").read_text()) if (MODEL / "config.json").is_file() else {}
    expected = {"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
                "intermediate_size": 14336, "num_attention_heads": 32,
                "num_key_value_heads": 8, "vocab_size": 128256}
    if any(model_config.get(key) != value for key, value in expected.items()):
        errors.append(f"model architecture mismatch: expected={expected}")
    if len(list((BUNDLE / "inputs/policies").glob("p[0-9][0-9].json"))) != 72:
        errors.append("policy input count is not 72")
    runtime_hits = {}
    for method in ("phase_hetero_mytest", "nvfp4_mytest", "sparse_bf16_mytest", "sparse_nvfp4_mytest", "marlin_nvfp4_mytest"):
        result = command_output(["rg", "-l", method, str(VLLM_ROOT / "vllm/vllm")])
        runtime_hits[method] = result["stdout"].splitlines()
        if result["returncode"] != 0:
            errors.append(f"patched vLLM method missing: {method}")
    python_checks = []
    for role, python, modules in (("cospaq", COSPAQ_PYTHON, "torch,transformers,datasets,numpy,pandas"),
                                  ("vllm", VLLM_PYTHON, "torch,transformers,vllm,bert_score,rouge_score,sacrebleu")):
        code = ("import importlib,json,sys; names='" + modules + "'.split(','); out={'executable':sys.executable};"
                "[(out.__setitem__(n,getattr(importlib.import_module(n),'__version__','ok'))) for n in names];print(json.dumps(out))")
        result = command_output([str(python), "-c", code])
        result["role"] = role
        python_checks.append(result)
        if result["returncode"]:
            errors.append(f"{role} interpreter import failed")
    git = {"fake": command_output(["git", "status", "--short"], cwd=REPO),
           "fake_commit": command_output(["git", "rev-parse", "HEAD"], cwd=REPO),
           "vllm": command_output(["git", "status", "--short"], cwd=VLLM_ROOT),
           "vllm_commit": command_output(["git", "rev-parse", "HEAD"], cwd=VLLM_ROOT),
           "vllm_diff": command_output(["git", "diff", "--binary"], cwd=VLLM_ROOT)}
    gpu = None
    if not args.no_gpu:
        gpu = command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
        if gpu["returncode"]:
            errors.append("nvidia-smi failed")
        elif any("RTX PRO 6000" not in line.upper() for line in gpu["stdout"].splitlines() if line.strip()):
            errors.append("visible GPUs are not all RTX Pro 6000")
    disk = shutil.disk_usage(BUNDLE)
    if disk.free < 250 * 2**30:
        errors.append("less than 250 GiB free")
    report = {"ok": not errors, "errors": errors, "paths": {k: str(v) for k, v in required.items()},
              "model": model_config, "runtime_hits": runtime_hits, "python": python_checks,
              "git": git, "gpu": gpu, "disk_free_gib": round(disk.free / 2**30, 2)}
    write_json(VALIDATION / ("preflight_no_gpu.json" if args.no_gpu else "preflight_gpu.json"), report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
