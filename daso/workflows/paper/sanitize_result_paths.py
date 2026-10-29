#!/usr/bin/env python3
"""Replace machine-local path prefixes in archived text results deterministically."""
from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS = (
    ("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf", "model://llama2-7b-chat"),
    ("/home/agent/wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct", "model://llama31-8b-instruct"),
    ("legacy-host://agent-wja/data/models/shakechen/Llama-2-7b-chat-hf", "model://llama2-7b-chat"),
    ("legacy-host://agent-wja/data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct", "model://llama31-8b-instruct"),
    ("/root/wja/project/my/cospaq/fake/", "legacy-artifact://cospaq-fake/"),
    ("/root/workspaces/cospaq/fake/", "legacy-artifact://cospaq-fake/"),
    ("/home/agent/wja/project/my/cospaq/fake/", "legacy-artifact://cospaq-fake/"),
    ("/home/agent/wja/project/my/cospaq/test/vllm/", "backend://vllm-cospaq/"),
    ("/root/wja/", "legacy-host://root-wja/"),
    ("/home/agent/wja/", "legacy-host://agent-wja/"),
    ("/root/workspaces/", "legacy-host://workspaces/"),
)
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".tex", ".toml", ".txt", ".yaml", ".yml"}


def sanitize(root: Path) -> tuple[int, int]:
    files = replacements = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        original = path.read_text()
        updated = original
        for source, target in REPLACEMENTS:
            updated = updated.replace(source, target)
        if updated != original:
            path.write_text(updated)
            files += 1
            replacements += sum(original.count(source) for source, _ in REPLACEMENTS)
    return files, replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/paper")
    args = parser.parse_args()
    files, replacements = sanitize(args.root.resolve())
    print(f"sanitized_files={files} replacements={replacements}")


if __name__ == "__main__":
    main()
