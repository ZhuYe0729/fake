#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

os.environ.setdefault("HF_HOME", "/home/agent/wja/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/agent/wja/.cache/huggingface/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/agent/wja/.cache/huggingface/transformers")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEBUG_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = FAKE_ROOT.parent
SOURCE_ROOT = FAKE_ROOT / "artifacts/results/main/003_llama2_7b_arc_easy_accuracy"
SOURCE_SCRIPTS = SOURCE_ROOT / "scripts"
PARETO_ROOT = FAKE_ROOT / "artifacts/debug/010_llama2_normal02_pareto_handoff"
PARETO_SCRIPTS = PARETO_ROOT / "scripts"

for path in (WORKSPACE_ROOT, FAKE_ROOT, SOURCE_SCRIPTS, PARETO_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (  # type: ignore  # noqa: E402
    DEFAULT_MODEL_KEY,
    METHODS,
    cleanup_cuda,
    compressible_modules,
    dtype_from_arg,
    load_model,
    load_tokenizer,
    model_result_root,
    model_spec,
    module_parent,
    replacement_report_dict,
    utc_now,
)
from eval import install_runtime_kernel  # type: ignore  # noqa: E402

PROMPT_TEMPLATE = "Summarize the following dialogue.\n\n{dialogue}\n\nSummary:"
UNIFORM_METHODS = (
    "dense_bf16",
    "dense_nvfp4",
    "sparse_bf16",
    "sparse_nvfp4",
    "marlin_nvfp4",
    "dense_nvfp4_prefill_marlin_decode",
)


def local_cuda_index(requested_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("CUDA is required")
    if requested_gpu < count:
        return requested_gpu
    if visible:
        return 0
    return requested_gpu


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def load_dialogsum_split(*, split: str, limit: int, cache_dir: str, local_files_only: bool):
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset("knkarthick/dialogsum", split=split, cache_dir=cache_dir, download_config=download_config)
    if limit is not None and limit >= 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def batch_items(items, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batch_count(num_items: int, batch_size: int) -> int:
    return (num_items + batch_size - 1) // batch_size


def load_compressed_state_into_model(model, *, method: str, source_root: Path, model_key: str, device: str) -> dict[str, Any] | None:
    if method == "dense_bf16":
        return None
    source_method = "dense_nvfp4" if method == "dense_nvfp4_prefill_marlin_decode" else method
    artifact = model_result_root(source_root, model_key) / "prepared" / source_method / "model.pt"
    metadata_path = artifact.parent / "metadata.json"
    log_path = artifact.parent / "compression_log.jsonl"
    if not artifact.exists():
        raise FileNotFoundError(f"missing prepared compressed artifact: {artifact}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing compression metadata: {metadata_path}")
    if source_method != "dense_bf16" and not log_path.exists():
        raise FileNotFoundError(f"missing compression log: {log_path}")
    payload = torch.load(artifact, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"failed to load compressed state: missing={missing} unexpected={unexpected}")
    model.to(device)
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source_method": source_method,
            "artifact": str(artifact),
            "metadata_path": str(metadata_path),
            "compression_log": str(log_path),
            "compression_log_exists": log_path.exists(),
        }
    )
    return metadata


def load_prepared_state(source_root: Path, method: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    artifact = source_root / "prepared" / method / "model.pt"
    metadata_path = artifact.parent / "metadata.json"
    log_path = artifact.parent / "compression_log.jsonl"
    if not artifact.exists():
        raise FileNotFoundError(f"missing prepared artifact for {method}: {artifact}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata for {method}: {metadata_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"missing compression log for {method}: {log_path}")
    payload = torch.load(artifact, map_location="cpu")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"prepared artifact for {method} has no state_dict")
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "source_method": method,
            "artifact": str(artifact),
            "metadata_path": str(metadata_path),
            "compression_log": str(log_path),
        }
    )
    return state, metadata


def policy_weight_source(prefill_backend: str, decode_backend: str) -> str | None:
    if prefill_backend != "dense_bf16":
        return prefill_backend
    if decode_backend != "dense_bf16":
        return decode_backend
    return None


def apply_policy_compressed_weights(model, policy: dict[str, Any], *, source_root: Path) -> dict[str, Any]:
    modules = {info.name: info for info in compressible_modules(model, DEFAULT_MODEL_KEY)}
    states: dict[str, dict[str, torch.Tensor]] = {}
    metadata: dict[str, Any] = {}
    replaced = 0
    expected = 0
    method_counts: Counter[str] = Counter()
    for item in policy["modules"]:
        module_name = item["module_name"]
        source_method = policy_weight_source(item["selected_prefill_backend"], item["selected_decode_backend"])
        if source_method is None:
            continue
        expected += 1
        method_counts[source_method] += 1
        if source_method not in states:
            states[source_method], metadata[source_method] = load_prepared_state(source_root, source_method)
        state = states[source_method]
        info = modules[module_name]
        key = f"{module_name}.weight"
        if key not in state:
            raise KeyError(f"{source_method} artifact missing {key}")
        parent, child_name = module_parent(model, info.name)
        module = getattr(parent, child_name)
        module.weight.data.copy_(state[key].to(device=module.weight.device, dtype=module.weight.dtype))
        bias_key = f"{module_name}.bias"
        if module.bias is not None and bias_key in state:
            module.bias.data.copy_(state[bias_key].to(device=module.bias.device, dtype=module.bias.dtype))
        replaced += 1
    if replaced != expected:
        raise RuntimeError(f"policy weight replacement mismatch: replaced={replaced} expected={expected}")
    return {
        "replaced_weight_modules": replaced,
        "expected_weight_modules": expected,
        "weight_source_counts": dict(method_counts),
        "source_metadata": metadata,
    }


def convert_policy_to_offline(policy_path: Path, out_path: Path) -> Path:
    from fake.kernels.offline_hybrid_policy import HybridPolicy, LayerPolicyDecision, save_policy_json

    payload = read_json(policy_path)
    decisions = []
    for item in payload["modules"]:
        decisions.append(
            LayerPolicyDecision(
                name=item["module_name"],
                n=int(item["n"]),
                k=int(item["k"]),
                count=1,
                selected_prefill_backend=item["selected_prefill_backend"],
                selected_decode_backend=item["selected_decode_backend"],
                selected_total_ms=float(item["selected_total_ms"]),
                selected_prefill_ms=float(item.get("selected_prefill_ms", 0.0)),
                selected_decode_ms=float(item.get("selected_decode_ms", 0.0)),
                selected_conversion_ms=float(item.get("selected_conversion_ms", 0.0)),
                strategy_candidates=[],
                prefill_candidates=[],
                decode_candidates=[],
                conversion_candidates=[],
                reason="dialogsum_pareto_quality_normal_02",
            )
        )
    scenario = payload["scenario"]
    policy = HybridPolicy(
        policy_format="offline_hybrid_policy_v1",
        scenario={
            "batch_size": int(scenario["batch_size"]),
            "input_tokens": int(scenario["input_tokens"]),
            "output_tokens": int(scenario["output_tokens"]),
            "m_prefill": int(scenario["m_prefill"]),
            "m_decode": int(scenario["m_decode"]),
        },
        kernels=["dense_bf16", "dense_nvfp4", "sparse_bf16", "sparse_nvfp4", "marlin_nvfp4"],
        include_conversion_cost=True,
        modules=decisions,
    )
    save_policy_json(policy, out_path)
    return out_path


def install_pareto_runtime(model, *, converted_policy_path: Path, dtype: torch.dtype) -> Any:
    from fake.models.llama_kernels import replace_linear_with_llama_predictor_hybrid

    report = replace_linear_with_llama_predictor_hybrid(
        model,
        policy_path=converted_policy_path,
        activation_dtype=dtype,
    )
    if int(report.skipped_linear_count) != 0:
        raise RuntimeError(f"pareto runtime skipped modules: {report.skipped[:5]}")
    return report


def install_uniform_runtime(model, *, method: str, model_key: str, dtype: torch.dtype) -> Any | None:
    if method == "dense_bf16":
        return None
    report = install_runtime_kernel(model, method, model_key=model_key, dtype=dtype)
    skipped = int(replacement_report_dict(report).get("skipped_linear_count", 0))
    if skipped != 0:
        raise RuntimeError(f"uniform runtime method={method} skipped modules: {replacement_report_dict(report).get('skipped', [])[:5]}")
    return report


def load_eval_model(*, dtype: torch.dtype, device: str, attn: str | None):
    from transformers import AutoModelForCausalLM

    spec = model_spec(DEFAULT_MODEL_KEY)
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": bool(spec["trust_remote_code"]),
    }
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(spec["path"], **kwargs)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    return model


@torch.inference_mode()
def dialogsum_generate_and_score(
    model,
    tokenizer,
    dataset,
    *,
    device: str,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    results_jsonl: Path,
) -> dict[str, Any]:
    existing = read_jsonl(results_jsonl)
    predictions = [str(row["prediction"]) for row in existing]
    references = [str(row["reference"]) for row in existing]
    count = len(existing)
    if count > len(dataset):
        raise RuntimeError(f"{results_jsonl} has {count} rows but dataset has {len(dataset)} samples")

    generation_config = GenerationConfig(
        max_length=None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    for batch in batch_items(dataset.select(range(count, len(dataset))) if hasattr(dataset, "select") else list(dataset)[count:], batch_size):
        dialogues = [str(example["dialogue"]) for example in batch]
        refs = [str(example["summary"]) for example in batch]
        prompts = [PROMPT_TEMPLATE.format(dialogue=dialogue) for dialogue in dialogues]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        generated = model.generate(**inputs, generation_config=generation_config)
        prompt_len = int(inputs["input_ids"].shape[1])
        preds = [tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True).strip() for tokens in generated]
        for example, prompt, pred, ref in zip(batch, prompts, preds, refs):
            row = {
                "id": example.get("id", example.get("fname", count)),
                "prompt": prompt,
                "dialogue": example["dialogue"],
                "prediction": pred,
                "reference": ref,
            }
            append_jsonl(results_jsonl, row)
            predictions.append(pred)
            references.append(ref)
            count += 1
        del inputs, generated
    rouge = compute_rouge(predictions, references)
    return {"num_samples": count, "rouge": rouge}


@torch.inference_mode()
def dialogsum_reference_nll(
    model,
    tokenizer,
    dataset,
    *,
    device: str,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    count = 0
    if batch_size != 1:
        raise ValueError("autoregressive DialogSum NLL currently requires batch_size=1")
    for batch in batch_items(dataset, batch_size):
        for example in batch:
            prompt = PROMPT_TEMPLATE.format(dialogue=str(example["dialogue"]))
            ref = str(example["summary"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_input_length).input_ids
            ref_ids = tokenizer(ref, add_special_tokens=False, truncation=True, max_length=max_target_length).input_ids
            if tokenizer.eos_token_id is not None:
                ref_ids = ref_ids + [tokenizer.eos_token_id]
            if not prompt_ids or not ref_ids:
                continue
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_tensor, use_cache=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].float()
            for target in ref_ids:
                target_tensor = torch.tensor([target], dtype=torch.long, device=device)
                loss = F.cross_entropy(next_logits, target_tensor, reduction="sum")
                total_loss += float(loss.item())
                total_tokens += 1
                token_input = target_tensor.reshape(1, 1)
                outputs = model(input_ids=token_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :].float()
                del target_tensor, loss, token_input
            count += 1
            del input_tensor, outputs, past_key_values, next_logits
    nll = total_loss / max(total_tokens, 1)
    return {"nll": nll, "ppl": float(math.exp(min(nll, 20.0))), "tokens": total_tokens, "loss_sum": total_loss, "num_samples": count}


def prepare_tokenizer():
    tokenizer = load_tokenizer(DEFAULT_MODEL_KEY)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return totals
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_for_rouge(pred)
        ref_tokens = tokenize_for_rouge(ref)
        totals["rouge1"] += rouge_n_f1(pred_tokens, ref_tokens, 1)
        totals["rouge2"] += rouge_n_f1(pred_tokens, ref_tokens, 2)
        totals["rougeL"] += rouge_l_f1(pred_tokens, ref_tokens)
    return {name: value / len(predictions) for name, value in totals.items()}


def tokenize_for_rouge(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rouge_n_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    pred_ngrams = ngrams(pred_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum((Counter(pred_ngrams) & Counter(ref_ngrams)).values())
    return f1_score(overlap, len(pred_ngrams), len(ref_ngrams))


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def rouge_l_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    return f1_score(lcs_len(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens))


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def f1_score(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    cleanup_cuda()


def runtime_report_dict(report: Any) -> dict[str, Any] | None:
    if report is None:
        return None
    out = replacement_report_dict(report)
    if hasattr(report, "backend_counts"):
        out["backend_counts"] = getattr(report, "backend_counts")
    return out
