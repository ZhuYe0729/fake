#!/usr/bin/env python3
"""Run the remaining prefill tasks through the frozen 042 evaluator."""
from __future__ import annotations

import importlib.util

from scenario import EVALUATOR


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_prefill_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TASKS = ("wikitext", "winogrande", "arc_easy", "mmlu")
    module.main()


if __name__ == "__main__":
    main()
