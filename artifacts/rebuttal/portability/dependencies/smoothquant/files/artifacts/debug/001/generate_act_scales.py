"""Generate SmoothQuant activation scales for llama-2-7b.

Uses cached wikitext-2-raw-v1 dataset (arrow format) for calibration
since the Pile dataset is unavailable in offline mode.
"""

import sys
import os
import functools
import argparse

import torch
import pyarrow as pa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

WIKITEXT_ARROW_PATH = (
    "/home/agent/wja/.cache/huggingface/Salesforce___wikitext"
    "/wikitext-2-raw-v1/0.0.0"
    "/b08601e04326c79dfdd32d625aee71d232d685c3"
)


def load_wikitext_texts(split="train"):
    """Load wikitext-2-raw-v1 text from cached arrow files."""
    arrow_file = os.path.join(WIKITEXT_ARROW_PATH, f"wikitext-{split}.arrow")
    table = pa.ipc.open_stream(arrow_file).read_all()
    texts = table.column("text").to_pylist()
    return [t for t in texts if t.strip()]


def get_act_scales(model, tokenizer, texts, num_samples=512, seq_len=512):
    """Collect per-channel activation max scales via forward hooks."""
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        coming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], coming_max)
        else:
            act_scales[name] = coming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    for i in tqdm(range(min(num_samples, len(texts))), desc="Calibrating"):
        input_ids = tokenizer(
            texts[i], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/agent/wja/data/models/LLM-Research/llama-2-7b",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Defaults to artifacts/debug/001/act_scales/llama-2-7b.pt",
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_path is None:
        args.output_path = os.path.join(
            os.path.dirname(__file__), "act_scales", "llama-2-7b.pt"
        )

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )

    print("Loading wikitext-2 calibration data...")
    texts = load_wikitext_texts("train")
    print(f"  {len(texts)} non-empty texts available")

    print(f"Generating act_scales with {args.num_samples} samples, seq_len={args.seq_len}...")
    act_scales = get_act_scales(
        model, tokenizer, texts, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)
    print(f"Saved act_scales to {args.output_path}")
    print(f"  Keys: {list(act_scales.keys())[:5]}...")
    print(f"  Total keys: {len(act_scales)}")


if __name__ == "__main__":
    main()
