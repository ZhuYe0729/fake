"""FP16/bfloat16 baseline accuracy evaluation on ARC tasks."""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

MODEL_PATH = "/home/agent/wja/data/models/LLM-Research/llama-2-7b"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "baseline.json")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )

    print("Creating HFLM wrapper...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)

    print("Evaluating on arc_easy and arc_challenge...")
    results = simple_evaluate(model=lm, tasks=["arc_easy", "arc_challenge"])

    # Extract key metrics
    summary = {}
    for task_name in ["arc_easy", "arc_challenge"]:
        if task_name in results.get("results", {}):
            task_results = results["results"][task_name]
            summary[task_name] = {
                "acc": task_results.get("acc,none"),
                "acc_norm": task_results.get("acc_norm,none"),
            }

    print("Baseline results:")
    for task, metrics in summary.items():
        print(f"  {task}: acc={metrics['acc']:.4f}, acc_norm={metrics['acc_norm']:.4f}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"baseline": summary, "full_results": str(results)}, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
