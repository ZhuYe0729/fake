"""FP8 E4M3 SmoothQuant test on llama-2-7b with arc_easy and arc_challenge.

Strategy: Try real FP8 inference (torch._scaled_mm RowWise) first.
If that fails or produces bad results, fall back to fake quant
(W8A8Fp8Linear simulated in bfloat16).
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add smoothquant repo to path
sys.path.insert(0, "/root/wja/project/my/cospaq/fake/third_party/smoothquant")
from smoothquant.smooth import smooth_lm

# Add this directory to path for local modules
sys.path.insert(0, os.path.dirname(__file__))
from fp8_real_linear import quantize_llama_like_real_fp8
from fp8_fake_quant import quantize_llama_like_fp8

MODEL_PATH = "/home/agent/wja/data/models/LLM-Research/llama-2-7b"
ACT_SCALES_PATH = os.path.join(os.path.dirname(__file__), "act_scales", "llama-2-7b.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "fp8.json")
ALPHA = 0.85


def try_real_fp8():
    """Attempt real FP8 E4M3 SmoothQuant inference via torch._scaled_mm."""
    print("=" * 60)
    print("FP8 E4M3 SmoothQuant — REAL FP8 attempt")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )

    act_scales = torch.load(ACT_SCALES_PATH)
    smooth_lm(model, act_scales, ALPHA)
    quantize_llama_like_real_fp8(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )

    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = simple_evaluate(model=lm, tasks=["arc_easy", "arc_challenge"])
    return results, "real_fp8"


def try_fake_fp8():
    """Fall back to fake FP8 E4M3 SmoothQuant (simulated in bfloat16)."""
    print("=" * 60)
    print("FP8 E4M3 SmoothQuant — Fake quant fallback")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )

    act_scales = torch.load(ACT_SCALES_PATH)
    smooth_lm(model, act_scales, ALPHA)
    quantize_llama_like_fp8(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=True,
    )

    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32)
    results = simple_evaluate(model=lm, tasks=["arc_easy", "arc_challenge"])
    return results, "fake_fp8"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Try real FP8 first
    results = None
    mode = None
    try:
        results, mode = try_real_fp8()
    except Exception as e:
        print(f"Real FP8 failed: {type(e).__name__}: {e}")
        print("Falling back to fake FP8...")
        results, mode = try_fake_fp8()

    # Extract key metrics
    summary = {}
    for task_name in ["arc_easy", "arc_challenge"]:
        if task_name in results.get("results", {}):
            task_results = results["results"][task_name]
            summary[task_name] = {
                "acc": task_results.get("acc,none"),
                "acc_norm": task_results.get("acc_norm,none"),
            }

    print(f"FP8 results ({mode}):")
    for task, metrics in summary.items():
        print(f"  {task}: acc={metrics['acc']:.4f}, acc_norm={metrics['acc_norm']:.4f}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"mode": mode, "summary": summary, "full_results": str(results)}, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
