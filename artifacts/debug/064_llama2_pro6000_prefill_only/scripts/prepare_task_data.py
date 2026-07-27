#!/usr/bin/env python3
"""Download/cache the five evaluation task datasets before offline execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(os.environ.get(
        "COSPAQ_TASK_CACHE", Path(__file__).resolve().parents[1] / "cache/huggingface")))
    parser.add_argument("--use-local-proxy", action="store_true")
    args = parser.parse_args()
    if args.use_local_proxy:
        os.environ["http_proxy"] = "http://127.0.0.1:8848"
        os.environ["https_proxy"] = "http://127.0.0.1:8848"
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:8848"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8848"
        # huggingface_hub uses httpx.  The vLLM environment does not install
        # the optional socksio package, so route HTTPS through the HTTP proxy
        # and remove a parent-shell SOCKS setting if one exists.
        os.environ.pop("all_proxy", None)
        os.environ.pop("ALL_PROXY", None)
        # The machine-wide mirror omits metadata required by the installed
        # huggingface_hub client.  The user-provided proxy reaches the official
        # endpoint, so make that endpoint explicit before importing datasets.
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str(args.cache)
    os.environ["HF_DATASETS_CACHE"] = str(args.cache / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from datasets import get_dataset_config_names, load_dataset
    mmlu_configs = tuple(config for config in get_dataset_config_names(
        "cais/mmlu", cache_dir=str(args.cache / "datasets"))
        if config not in {"all", "auxiliary_train"})
    if len(mmlu_configs) != 57:
        raise RuntimeError(f"expected 57 MMLU subject configs, found {len(mmlu_configs)}")
    requests = (
        # Match the dataset_path/dataset_name frozen by this checkout's
        # lm-eval task YAML, rather than the token-level calibration source.
        ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1"),
        ("allenai/winogrande", "winogrande_xl"),
        ("allenai/ai2_arc", "ARC-Easy"),
        ("allenai/ai2_arc", "ARC-Challenge"),
    ) + tuple(("cais/mmlu", config) for config in mmlu_configs)
    rows = []
    # Dataset payloads may live in a shared machine cache, but experiment
    # provenance always remains in the isolated bundle.
    output = Path(__file__).resolve().parents[1] / "cache/task_data_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for dataset, config in requests:
        try:
            loaded = load_dataset(dataset, config, cache_dir=str(args.cache / "datasets"))
            rows.append({"dataset": dataset, "config": config, "status": "complete",
                         "splits": {name: len(split) for name, split in loaded.items()}})
        except Exception as exc:
            rows.append({"dataset": dataset, "config": config, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})
            output.write_text(json.dumps({"complete": False,
                              "used_local_proxy": args.use_local_proxy,
                              "datasets": rows}, indent=2, sort_keys=True) + "\n")
            raise
        output.write_text(json.dumps({"complete": False,
                          "used_local_proxy": args.use_local_proxy,
                          "datasets": rows}, indent=2, sort_keys=True) + "\n")
    output.write_text(json.dumps({"complete": True,
                      "used_local_proxy": args.use_local_proxy,
                      "datasets": rows}, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
