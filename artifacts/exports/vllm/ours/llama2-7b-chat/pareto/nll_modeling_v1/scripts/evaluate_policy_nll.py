#!/usr/bin/env python3
"""Teacher-force PMPD NLL for phase policies using the prepared compression weights."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf")
PREPARED = Path("/home/agent/wja/project/my/cospaq/fake/artifacts/exports/vllm/baselines/llama2-7b-chat/prepared")
METHOD_STATE = {"dense_nvfp4": "dense_nvfp4", "sparse_bf16": "sparse_bf16", "sparse_nvfp4": "sparse_nvfp4", "w4a16_ours": "marlin_nvfp4"}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=("prefill_only", "prefill_decode"), required=True)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--prepared-root", type=Path, default=PREPARED)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--decode-tokens", type=int, default=80)
    p.add_argument("--max-samples", type=int, default=0, help="debug cap; 0 keeps all 300 calibration samples")
    p.add_argument("--policy", default="", help="one policy id, otherwise evaluate all incomplete policies")
    p.add_argument("--output-csv", type=Path, default=None, help="per-policy shard path; enables parallel workers")
    return p.parse_args()


def parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    obj = model
    for part in name.split(".")[:-1]: obj = getattr(obj, part)
    return obj, name.rsplit(".", 1)[-1]


def source_names(fused_name: str) -> list[str]:
    base = fused_name.rsplit(".", 1)[0]
    typ = fused_name.rsplit(".", 1)[-1]
    if typ == "qkv_proj": return [base + ".q_proj", base + ".k_proj", base + ".v_proj"]
    if typ == "gate_up_proj": return [base + ".gate_proj", base + ".up_proj"]
    return [fused_name]


def install(model: nn.Module, policy: dict, phase: str, prepared_root: Path) -> list[tuple[nn.Module, str, nn.Module]]:
    saved = []
    # A full prepared `.pt` state is ~13 GB. Load one method at a time rather
    # than retaining all four in host memory during a 7B forward pass.
    for method in METHOD_STATE:
        selected = [fused for fused, entry in policy["method_map"].items() if entry[f"{phase}_method"] == method]
        if not selected: continue
        state = torch.load(prepared_root / METHOD_STATE[method] / "model.pt", map_location="cpu")["state_dict"]
        for fused in selected:
            for name in source_names(fused):
                obj, child = parent(model, name); old = getattr(obj, child)
                if not isinstance(old, nn.Linear): raise TypeError(f"{name}: {type(old)}")
                replacement = nn.Linear(old.in_features, old.out_features, bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
                replacement.weight.data.copy_(state[f"{name}.weight"].to(device=old.weight.device, dtype=old.weight.dtype))
                if old.bias is not None: replacement.bias.data.copy_(old.bias.data)
                setattr(obj, child, replacement); saved.append((obj, child, old))
        del state; gc.collect()
    return saved


def restore(saved: Iterable[tuple[nn.Module, str, nn.Module]]) -> None:
    for obj, child, old in saved: setattr(obj, child, old)


def token_batches(rows: list[dict], tok, max_prompt: int, decode_tokens: int, batch: int, target: str):
    encoded = []
    for row in rows:
        prompt = tok.encode(row["prompt"], add_special_tokens=True)[-max_prompt:]
        reference = tok.encode(row["reference"], add_special_tokens=False)[:decode_tokens]
        ids = prompt if target == "prefill" else prompt + reference
        labels = list(ids)
        if target == "decode": labels[:len(prompt)] = [-100] * len(prompt)
        if len(ids) > 1: encoded.append((ids, labels))
    for start in range(0, len(encoded), batch):
        group = encoded[start:start + batch]; width = max(len(x[0]) for x in group)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(group), width), pad, dtype=torch.long); labels = torch.full_like(ids, -100)
        for i, (tokens, target_labels) in enumerate(group): ids[i, :len(tokens)] = torch.tensor(tokens); labels[i, :len(tokens)] = torch.tensor(target_labels)
        yield ids, labels


@torch.inference_mode()
def nll(model, rows, tok, device, max_prompt, decode_tokens, batch, target) -> dict:
    loss_sum = tokens = 0
    for ids, labels in token_batches(rows, tok, max_prompt, decode_tokens, batch, target):
        logits = model(input_ids=ids.to(device), use_cache=False).logits[:, :-1].float()
        labels = labels[:, 1:].to(device)
        loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum").item())
        tokens += int((labels != -100).sum().item())
    return {"nll": loss_sum / max(tokens, 1), "loss_sum": loss_sum, "tokens": tokens}


def read_rows(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text().splitlines() if line]
def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []
def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f: w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    a = parse(); torch.cuda.set_device(a.gpu); device = f"cuda:{a.gpu}"
    rows = read_rows(a.output_root / "samples" / "pmpd_100x3.jsonl")
    if a.max_samples: rows = rows[:a.max_samples]
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True); tok.pad_token = tok.pad_token or tok.eos_token
    # The installed FlashAttention binary can terminate the interpreter on an
    # RTX 5090 before Python receives an exception. NLL calibration is an
    # offline measurement, so the portable eager attention path is preferred.
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(device).eval()
    output = a.output_csv or a.output_root / "nll" / f"{a.scenario}.csv"; result = read_csv(output)
    done = {r["policy_id"] for r in result if int(r.get("sample_count", "0")) == len(rows)}
    print(f"measuring dense reference on {len(rows)} samples", flush=True)
    dense_prefill = nll(model, rows, tok, device, a.max_prompt_tokens, a.decode_tokens, a.batch_size, "prefill")
    dense_decode = nll(model, rows, tok, device, a.max_prompt_tokens, a.decode_tokens, a.batch_size, "decode")
    policies = sorted((a.output_root / "policies" / a.scenario).glob("p*.json"))
    for path in policies:
        item = json.loads(path.read_text()); pid = item["policy_id"]
        if pid in done or (a.policy and pid != a.policy): continue
        saved = install(model, item, "prefill", a.prepared_root)
        try: measured_prefill = nll(model, rows, tok, device, a.max_prompt_tokens, a.decode_tokens, a.batch_size, "prefill")
        finally: restore(saved); torch.cuda.empty_cache()
        if a.scenario == "prefill_decode":
            saved = install(model, item, "decode", a.prepared_root)
            try: measured_decode = nll(model, rows, tok, device, a.max_prompt_tokens, a.decode_tokens, a.batch_size, "decode")
            finally: restore(saved); torch.cuda.empty_cache()
        else: measured_decode = dense_decode
        delta_prefill = measured_prefill["nll"] - dense_prefill["nll"]; delta_decode = measured_decode["nll"] - dense_decode["nll"]
        result = [r for r in result if r["policy_id"] != pid]
        result.append({"policy_id": pid, "scenario": a.scenario, "sample_count": len(rows), "dense_prefill_nll": dense_prefill["nll"], "dense_decode_nll": dense_decode["nll"], "prefill_nll": measured_prefill["nll"], "decode_nll": measured_decode["nll"], "delta_prefill_nll": delta_prefill, "delta_decode_nll": delta_decode, "target_delta_nll": delta_prefill + (80 * delta_decode if a.scenario == "prefill_decode" else 0.0), "prefill_tokens": measured_prefill["tokens"], "decode_tokens": measured_decode["tokens"]})
        write_csv(output, result); print(f"finished {a.scenario}/{pid}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__": main()
