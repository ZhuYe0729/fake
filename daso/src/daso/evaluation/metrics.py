from __future__ import annotations

from typing import Any


METRIC_KEYS = {
    "boolq": "acc",
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "piqa": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
    "wikitext": "word_perplexity",
    "cnn_dailymail": "rougeL",
    "dialogsum": "rougeL",
    "iwslt2017": "sacrebleu",
}


def select_metric(task: str, result: dict[str, Any]) -> float:
    key = METRIC_KEYS[task]
    current: Any = result
    if "results" in current and task in current["results"]:
        current = current["results"][task]
    if key not in current:
        candidates = [name for name in current if name.split(",", 1)[0] == key]
        if len(candidates) != 1:
            raise KeyError(f"cannot find metric {key} for {task}")
        key = candidates[0]
    return float(current[key])

