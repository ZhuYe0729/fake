#!/usr/bin/env python3
"""Qwen3.5 prefill/decode speed benchmark with module-level time breakdown.

This script measures text-only and multimodal (image+text) performance of Qwen3.5
models, providing:
  - Prefill latency (ms)
  - Decode latency (ms/token, steady-state) using KV cache
  - First decode step latency (ms)
  - Coarse breakdown: hybrid_linear_attn_block, full_attn_block, mlp_block, norm,
    lm_head, vision, all_linear, other (percentages)
  - Fine breakdown: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
    attn_core, activation, norm, lm_head (percentages)

Speed ≠ Breakdown: default (no --breakdown) measures real latency without hooks;
--breakdown runs separately with hooks and is only for time-ratio analysis.

nsys cross-validation command (not integrated):
  nsys profile --trace=cuda,nvtx,cublas,cudnn -o qwen_profile \\
    python scripts/bench_qwen3_5_speed.py --input-tokens 1024 --output-tokens 128

TODO model sizes (pass via --model-path):
  /home/agent/wja/data/models/Qwen/Qwen3.5-2B
  /home/agent/wja/data/models/Qwen/Qwen3.5-4B
  /home/agent/wja/data/models/Qwen/Qwen3.5-9B
  /home/agent/wja/data/models/Qwen/Qwen3.5-27B
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fake.models.qwen3_5 import DEFAULT_QWEN3_5_VARIANT, QWEN3_5_VARIANTS, qwen3_5_model_path
from fake.models.qwen3_5_kernels import (
    QWEN3_5_REAL_KERNEL_METHODS,
    default_qwen3_5_kernel_checkpoint_path,
    load_qwen3_5_kernel_checkpoint_into_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(qwen3_5_model_path(DEFAULT_QWEN3_5_VARIANT))
DEFAULT_OUTPUT_CSV = "artifacts/results/qwen3_5_speed.csv"
DEFAULT_INPUT_TOKENS = [128, 512, 1024, 2048]
DEFAULT_OUTPUT_TOKENS = [64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]

COARSE_MODULE_LABELS = [
    "hybrid_linear_attn_block",
    "full_attn_block",
    "mlp_block",
    "norm",
    "lm_head",
    "vision",
    "all_linear",
    "other",
]

FINE_MODULE_LABELS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "attn_core",
    "hybrid_attn_core",
    "hybrid_conv1d",
    "activation",
    "norm",
    "lm_head",
    "vision",
    "all_linear",
    "other",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
    multimodal: bool,
    device_map: str | None = None,
    max_memory: dict[int | str, str] | None = None,
):
    """Load Qwen3.5 model. Uses CausalLM for text-only, ConditionalGeneration for multimodal."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    load_kwargs = dict(
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
    )
    if device_map is not None:
        load_kwargs["device_map"] = device_map
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

    if multimodal:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **load_kwargs,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )
    model.eval()
    return model


def load_processor(model_path: str):
    """Load processor for multimodal input preparation."""
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------


def classify_layers(model) -> dict:
    """Walk the model and classify each decoder layer and vision block.

    Returns a dict with:
      - decoder_layers: list of dicts with index, module, type (full/hybrid), class_name
      - vision_blocks: list of dicts with index, module, class_name
      - num_full_attn: int
      - num_hybrid_linear_attn: int
      - num_mlp: int
      - num_vision_blocks: int
    """
    result = {"decoder_layers": [], "vision_blocks": []}

    # Find decoder layers
    lm = _get_language_model(model)
    layers = lm.layers if hasattr(lm, "layers") else getattr(lm, "decoder_layers", None)
    if layers is None:
        raise RuntimeError("Cannot find decoder layers in language model")

    num_full = 0
    num_hybrid = 0
    num_mlp = 0

    for i, layer in enumerate(layers):
        layer_class = type(layer).__name__
        has_self_attn = hasattr(layer, "self_attn")
        has_linear_attn = hasattr(layer, "linear_attn")
        has_mlp = hasattr(layer, "mlp")

        if has_self_attn:
            layer_type = "full_attention"
            num_full += 1
        elif has_linear_attn:
            layer_type = "hybrid_linear_attention"
            num_hybrid += 1
        else:
            layer_type = "unknown"

        if has_mlp:
            num_mlp += 1

        result["decoder_layers"].append(
            {
                "index": i,
                "module": layer,
                "type": layer_type,
                "class_name": layer_class,
            }
        )

    # Find vision blocks
    visual = _get_visual(model)
    if visual is not None and hasattr(visual, "blocks"):
        for i, block in enumerate(visual.blocks):
            result["vision_blocks"].append(
                {
                    "index": i,
                    "module": block,
                    "class_name": type(block).__name__,
                }
            )

    result["num_full_attn"] = num_full
    result["num_hybrid_linear_attn"] = num_hybrid
    result["num_mlp"] = num_mlp
    result["num_vision_blocks"] = len(result["vision_blocks"])

    return result


def _get_language_model(model) -> nn.Module:
    """Get the text/language model from any Qwen3.5 wrapper."""
    # Qwen3_5ForConditionalGeneration: model.model.language_model
    # Qwen3_5ForCausalLM: model.model (which IS Qwen3_5TextModel)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            return inner.language_model
        # Qwen3_5ForCausalLM: inner IS Qwen3_5TextModel
        if hasattr(inner, "layers"):
            return inner
    # Direct Qwen3_5Model
    if hasattr(model, "language_model"):
        return model.language_model
    raise RuntimeError(f"Cannot find language model in {type(model).__name__}")


def _get_visual(model) -> nn.Module | None:
    """Get the vision model if present."""
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    return None


def print_model_summary(model, layer_info: dict) -> None:
    """Print model architecture summary for verification."""
    print(f"Model: {type(model).__name__}")
    full_indices = [d["index"] for d in layer_info["decoder_layers"] if d["type"] == "full_attention"]
    hybrid_indices = [d["index"] for d in layer_info["decoder_layers"] if d["type"] == "hybrid_linear_attention"]
    print(f"Full attention layers: {layer_info['num_full_attn']} (indices: {full_indices})")
    print(f"Hybrid linear attention layers: {layer_info['num_hybrid_linear_attn']} (indices: {hybrid_indices})")
    print(f"Vision encoder blocks: {layer_info['num_vision_blocks']}")
    print(f"MLP blocks: {layer_info['num_mlp']}")

    # Print per-layer class names for verification
    for d in layer_info["decoder_layers"][:3]:
        print(f"  Layer {d['index']}: {d['class_name']} -> {d['type']}")
    if layer_info["decoder_layers"]:
        # Also print a full_attention layer if one exists after the first 3
        fa_layers = [d for d in layer_info["decoder_layers"] if d["type"] == "full_attention"]
        if fa_layers:
            fa = fa_layers[0]
            print(f"  Layer {fa['index']}: {fa['class_name']} -> {fa['type']}")
            # Print submodules
            for name, _mod in fa["module"].named_children():
                print(f"    {name}: {type(_mod).__name__}")

    if layer_info["num_vision_blocks"] > 0:
        vb = layer_info["vision_blocks"][0]
        print(f"  Vision block 0: {vb['class_name']}")
        for name, _mod in vb["module"].named_children():
            print(f"    {name}: {type(_mod).__name__}")


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def prepare_text_inputs(
    model,
    batch_size: int,
    input_tokens: int,
    mode: str = "random",
    prompt_text: str | None = None,
):
    """Prepare batched token inputs for text-only benchmark.

    Args:
        model: The loaded model.
        batch_size: Number of sequences.
        input_tokens: Number of input tokens per sequence.
        mode: 'random', 'repeat', or 'file'.
        prompt_text: Text to repeat (for 'repeat' mode) or file path (for 'file' mode).

    Returns:
        input_ids: LongTensor of shape (batch_size, input_tokens) on the model's device.
    """
    device = next(model.parameters()).device

    if mode == "random":
        vocab_size = _get_vocab_size(model)
        input_ids = torch.randint(0, min(vocab_size, 100000), (batch_size, input_tokens), device=device)
        return input_ids

    if mode == "repeat":
        if prompt_text is None:
            prompt_text = "The quick brown fox jumps over the lazy dog. "
        # Get tokenizer
        tokenizer = _get_tokenizer(model)
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if not tokens:
            raise ValueError("Prompt text produced no tokens")
        # Repeat to reach input_tokens
        repeated = (tokens * ((input_tokens // len(tokens)) + 1))[:input_tokens]
        input_ids = torch.tensor([repeated] * batch_size, device=device)
        return input_ids

    if mode == "file":
        if prompt_text is None:
            raise ValueError("--prompt-file required for input-mode=file")
        text = Path(prompt_text).read_text()
        tokenizer = _get_tokenizer(model)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < input_tokens:
            tokens = (tokens * ((input_tokens // len(tokens)) + 1))[:input_tokens]
        else:
            tokens = tokens[:input_tokens]
        input_ids = torch.tensor([tokens] * batch_size, device=device)
        return input_ids

    raise ValueError(f"Unknown input mode: {mode}")


def prepare_multimodal_inputs(
    model,
    processor,
    batch_size: int,
    input_tokens: int,
    image_path: str,
    mode: str = "random",
    prompt_text: str | None = None,
) -> dict:
    """Prepare image+text inputs.

    Returns a dict of model inputs including pixel_values etc.
    Also returns metadata: text_tokens, image_tokens, total_prefill_tokens, image_size.
    """
    from PIL import Image

    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    image_size = image.size  # (width, height)

    # Prepare text part
    if mode == "repeat" and prompt_text:
        text = prompt_text
    else:
        text = "Describe this image in detail."

    # Build messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    # Use processor: pass messages + images together so image placeholders
    # are correctly inserted into the token sequence.
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Estimate token counts
    input_ids = inputs["input_ids"]
    total_tokens = input_ids.shape[1]

    # Count image tokens from image_grid_thw if available
    image_tokens_est = None
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"]
        if isinstance(thw, torch.Tensor) and thw.numel() > 0:
            # image_grid_thw shape: (num_images, 3) with (T, H, W) in patches
            # Actual visual tokens per image = T * H * W / (spatial_merge^2)
            visual = _get_visual(model)
            merge_size = 2
            if visual is not None and hasattr(visual, 'config'):
                merge_size = getattr(visual.config, 'spatial_merge_size', 2)
            image_tokens_est = int((thw[:, 0] * thw[:, 1] * thw[:, 2] / (merge_size ** 2)).sum().item())

    text_tokens = total_tokens  # Note: includes image placeholder tokens in the sequence

    # For batch > 1, we'd need to duplicate. For simplicity, we support batch=1 for multimodal.
    if batch_size > 1:
        print(f"  Warning: multimodal mode currently only supports batch_size=1, got {batch_size}. Using 1.")
        batch_size = 1

    metadata = {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens_est,
        "total_prefill_tokens": total_tokens,
        "image_size": image_size,
    }

    return inputs, metadata


def _get_vocab_size(model) -> int:
    lm = _get_language_model(model)
    if hasattr(lm, "embed_tokens"):
        return lm.embed_tokens.weight.shape[0]
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.shape[0]
    # Fallback - Qwen3.5 vocab
    return 248320


def _get_tokenizer(model):
    """Get a tokenizer from the model's config."""
    from transformers import AutoTokenizer

    model_path = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", None)
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Speed benchmark (no hooks)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def benchmark_prefill_decode(
    model,
    input_ids: torch.Tensor,
    output_tokens: int,
    warmup: int,
    iters: int,
    sampling: bool = False,
) -> dict:
    """Measure prefill and decode latency using KV cache. No hooks.

    Returns dict with:
      prefill_ms, decode_total_ms, first_decode_ms, decode_per_token_ms, tokens_per_sec
    """
    batch_size = input_ids.shape[0]

    # Warmup
    for _ in range(warmup):
        _do_one_generate_cycle(model, input_ids, output_tokens, sampling)

    torch.cuda.synchronize()

    prefill_times: list[float] = []
    decode_times: list[list[float]] = []  # [iter][step]
    first_decode_times: list[float] = []
    decode_step_times_flat: list[float] = []  # all decode steps except first

    for _ in range(iters):
        prefill_ms, decode_step_ms_list = _do_one_generate_cycle(model, input_ids, output_tokens, sampling)
        prefill_times.append(prefill_ms)
        decode_times.append(decode_step_ms_list)
        if decode_step_ms_list:
            first_decode_times.append(decode_step_ms_list[0])
            decode_step_times_flat.extend(decode_step_ms_list[1:] if len(decode_step_ms_list) > 1 else [])

    mean_prefill_ms = statistics.fmean(prefill_times)
    mean_first_decode_ms = statistics.fmean(first_decode_times) if first_decode_times else 0.0
    mean_decode_per_token_ms = statistics.fmean(decode_step_times_flat) if decode_step_times_flat else 0.0
    total_decode_ms = sum(decode_step_times_flat) / len(decode_times) if decode_times else 0.0
    # Tokens per second = batch_size * tokens_per_second (for decode)
    tokens_per_sec = batch_size * 1000.0 / mean_decode_per_token_ms if mean_decode_per_token_ms > 0 else 0.0

    return {
        "prefill_ms": mean_prefill_ms,
        "decode_total_ms": total_decode_ms if decode_step_times_flat else sum(first_decode_times) / len(first_decode_times) if first_decode_times else 0.0,
        "first_decode_ms": mean_first_decode_ms,
        "decode_per_token_ms": mean_decode_per_token_ms,
        "tokens_per_sec": tokens_per_sec,
    }


def _do_one_generate_cycle(model, input_ids, output_tokens, sampling):
    """Run one prefill + full decode cycle. Returns (prefill_ms, list[decode_ms_per_step])."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # --- Prefill ---
    start_ev.record()
    outputs = model(input_ids=input_ids, use_cache=True)
    end_ev.record()
    torch.cuda.synchronize()
    prefill_ms = start_ev.elapsed_time(end_ev)

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1:, :]

    decode_step_times: list[float] = []
    d_start = torch.cuda.Event(enable_timing=True)
    d_end = torch.cuda.Event(enable_timing=True)

    for _step in range(output_tokens):
        if sampling:
            next_token = torch.multinomial(
                torch.softmax(next_logits.squeeze(1).float(), dim=-1), num_samples=1
            ).unsqueeze(0)
        else:
            next_token = next_logits.argmax(dim=-1)

        d_start.record()
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        d_end.record()
        torch.cuda.synchronize()
        decode_step_times.append(d_start.elapsed_time(d_end))

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

    return prefill_ms, decode_step_times


# ---------------------------------------------------------------------------
# Breakdown benchmark (with hooks)
# ---------------------------------------------------------------------------


class _HookTimingCollector:
    """Collects timing from forward hooks using CUDA events.

    Uses per-hook closure variables (not module attributes) to avoid collisions
    when multiple hooks are registered on the same module for different labels.
    """

    def __init__(self):
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def register(self, module: nn.Module, label: str):
        """Register pre/post hooks that record CUDA events.

        Each hook pair stores its start event in a closure cell, so multiple
        labels on the same module don't interfere.
        """
        start_holder: list[torch.cuda.Event | None] = [None]

        def pre_hook(_mod, _inp):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            start_holder[0] = start

        def post_hook(_mod, _inp, _out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if start_holder[0] is not None:
                self.events[label].append((start_holder[0], end))

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def collect_all(self) -> dict[str, float]:
        """Synchronize and return aggregated times in ms (sum across all hook invocations)."""
        torch.cuda.synchronize()
        result = {}
        for label, pairs in self.events.items():
            total = sum(s.elapsed_time(e) for s, e in pairs)
            result[label] = total
        self.events.clear()
        return result

    def drain(self) -> None:
        """Discard all accumulated events without synchronizing (call after sync)."""
        self.events.clear()


def install_coarse_hooks(model, layer_info: dict, collector: _HookTimingCollector):
    """Install hooks for coarse breakdown: hybrid_linear_attn_block, full_attn_block, mlp_block, norm, lm_head, vision.

    Coarse mode hooks parent blocks (inclusive timing). Does NOT hook children of hooked modules.
    """
    lm = _get_language_model(model)

    # Hook per-layer blocks
    for d in layer_info["decoder_layers"]:
        layer = d["module"]
        layer_idx = d["index"]
        if d["type"] == "full_attention" and hasattr(layer, "self_attn"):
            collector.register(layer.self_attn, "full_attn_block")
        elif d["type"] == "hybrid_linear_attention" and hasattr(layer, "linear_attn"):
            collector.register(layer.linear_attn, "hybrid_linear_attn_block")
        if hasattr(layer, "mlp"):
            collector.register(layer.mlp, "mlp_block")
        if hasattr(layer, "input_layernorm"):
            collector.register(layer.input_layernorm, "norm")
        if hasattr(layer, "post_attention_layernorm"):
            collector.register(layer.post_attention_layernorm, "norm")

    # Final norm in language model
    if hasattr(lm, "norm") and lm.norm is not None:
        collector.register(lm.norm, "norm")

    # lm_head
    if hasattr(model, "lm_head") and model.lm_head is not None:
        collector.register(model.lm_head, "lm_head")

    # Vision blocks
    visual = _get_visual(model)
    if visual is not None and hasattr(visual, "blocks"):
        for vb in layer_info["vision_blocks"]:
            collector.register(vb["module"], "vision")

    # Register hooks on ALL nn.Linear modules (for all_linear stats).
    # These are indexed by the coarse labels so won't double-count when computing
    # percentages - we use them only for the all_linear aggregate.
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            collector.register(mod, "_all_linear_internal")


def install_fine_hooks(model, layer_info: dict, collector: _HookTimingCollector):
    """Install hooks for fine breakdown.

    Hooks individual Linear sub-modules (q_proj, k_proj, v_proj, o_proj etc.),
    plus activation functions. Does NOT hook parent blocks to avoid double-counting.
    attn_core time = self_attn total - sum(q/k/v/o projection) for full attention layers.
    """
    lm = _get_language_model(model)

    for d in layer_info["decoder_layers"]:
        layer = d["module"]

        # Full attention layer: hook projections individually
        if d["type"] == "full_attention" and hasattr(layer, "self_attn"):
            sa = layer.self_attn
            if hasattr(sa, "q_proj"):
                collector.register(sa.q_proj, "q_proj")
            if hasattr(sa, "k_proj"):
                collector.register(sa.k_proj, "k_proj")
            if hasattr(sa, "v_proj"):
                collector.register(sa.v_proj, "v_proj")
            if hasattr(sa, "o_proj"):
                collector.register(sa.o_proj, "o_proj")
            # Hook the full self_attn as well to compute attn_core = self_attn - (q+k+v+o)
            collector.register(sa, "_full_attn_total")

        # Hybrid linear attention layer: hook projections individually
        elif d["type"] == "hybrid_linear_attention" and hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            if hasattr(la, "in_proj_qkv"):
                collector.register(la.in_proj_qkv, "in_proj_qkv")
            if hasattr(la, "in_proj_z"):
                collector.register(la.in_proj_z, "in_proj_z")
            if hasattr(la, "in_proj_a"):
                collector.register(la.in_proj_a, "in_proj_a")
            if hasattr(la, "in_proj_b"):
                collector.register(la.in_proj_b, "in_proj_b")
            if hasattr(la, "out_proj"):
                collector.register(la.out_proj, "out_proj")
            if hasattr(la, "act"):
                collector.register(la.act, "activation")
            if hasattr(la, "conv1d"):
                collector.register(la.conv1d, "hybrid_conv1d")
            if hasattr(la, "norm"):
                collector.register(la.norm, "norm")
            # Hook the full linear_attn to derive hybrid_attn_core
            collector.register(la, "_hybrid_attn_total")

        # MLP projections
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj"):
                collector.register(mlp.gate_proj, "gate_proj")
            if hasattr(mlp, "up_proj"):
                collector.register(mlp.up_proj, "up_proj")
            if hasattr(mlp, "down_proj"):
                collector.register(mlp.down_proj, "down_proj")
            if hasattr(mlp, "act_fn"):
                collector.register(mlp.act_fn, "activation")

        # Norms
        if hasattr(layer, "input_layernorm"):
            collector.register(layer.input_layernorm, "norm")
        if hasattr(layer, "post_attention_layernorm"):
            collector.register(layer.post_attention_layernorm, "norm")

    # Final norm
    if hasattr(lm, "norm") and lm.norm is not None:
        collector.register(lm.norm, "norm")

    # lm_head
    if hasattr(model, "lm_head") and model.lm_head is not None:
        collector.register(model.lm_head, "lm_head")

    # Vision blocks (fine)
    visual = _get_visual(model)
    if visual is not None and hasattr(visual, "blocks"):
        for vb in layer_info["vision_blocks"]:
            collector.register(vb["module"], "vision")

    # All nn.Linear
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            collector.register(mod, "_all_linear_internal")


@torch.inference_mode()
def benchmark_breakdown(
    model,
    input_ids: torch.Tensor,
    output_tokens: int,
    warmup: int,
    iters: int,
    breakdown_mode: str,
    layer_info: dict,
) -> dict:
    """Run breakdown benchmark with hooks installed.

    Returns separate prefill and decode breakdown dicts with module label -> pct.
    """
    collector = _HookTimingCollector()

    if breakdown_mode == "coarse":
        install_coarse_hooks(model, layer_info, collector)
    else:
        install_fine_hooks(model, layer_info, collector)

    # Warmup
    for _ in range(warmup):
        _do_one_breakdown_cycle(model, input_ids, output_tokens, collector)

    # Measurement iterations
    prefill_results: list[dict[str, float]] = []
    decode_results: list[dict[str, float]] = []

    for _ in range(iters):
        pref_agg, dec_agg = _do_one_breakdown_cycle(model, input_ids, output_tokens, collector)
        prefill_results.append(pref_agg)
        decode_results.append(dec_agg)

    # Average across iterations
    prefill_avg = _average_timing_dicts(prefill_results)
    decode_avg = _average_timing_dicts(decode_results)

    # Post-process: compute attn_core and all_linear
    if breakdown_mode == "fine":
        prefill_avg = _compute_fine_derived(prefill_avg)
        decode_avg = _compute_fine_derived(decode_avg)
    else:
        # In coarse mode, just rename _all_linear_internal to all_linear
        prefill_avg["all_linear"] = prefill_avg.pop("_all_linear_internal", 0.0)
        decode_avg["all_linear"] = decode_avg.pop("_all_linear_internal", 0.0)

    # Convert to percentages
    prefill_total = prefill_avg.get("_total_ms", sum(prefill_avg.values()))
    decode_total = decode_avg.get("_total_ms", sum(decode_avg.values()))

    prefill_pct = _to_pct(prefill_avg, prefill_total)
    decode_pct = _to_pct(decode_avg, decode_total)

    # Compute "other" for both phases
    _compute_other(prefill_pct, prefill_total, breakdown_mode)
    _compute_other(decode_pct, decode_total, breakdown_mode)

    # Merge prefill and decode results with prefixes
    result = {}
    for k, v in prefill_pct.items():
        result[f"prefill_{k}_pct"] = v
    for k, v in decode_pct.items():
        result[f"decode_{k}_pct"] = v
    result["prefill_total_ms"] = prefill_total
    result["decode_total_ms"] = decode_total

    return result


def _do_one_breakdown_cycle(model, input_ids, output_tokens, collector):
    """Run one prefill + decode cycle collecting hook timings. Returns (prefill_agg, decode_agg)."""
    # Total timing for prefill
    p_start = torch.cuda.Event(enable_timing=True)
    p_end = torch.cuda.Event(enable_timing=True)

    p_start.record()
    outputs = model(input_ids=input_ids, use_cache=True)
    p_end.record()

    prefill_raw = collector.collect_all()  # dict label -> ms
    torch.cuda.synchronize()
    prefill_raw["_total_ms"] = p_start.elapsed_time(p_end)

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1:, :]

    # Decode steps: run all decode steps but only collect events from the first
    # step (steady-state representative). Hook events from subsequent steps are
    # drained at the end to avoid contaminating the next iteration.
    decode_raw: dict[str, float] = {}
    d_start = torch.cuda.Event(enable_timing=True)
    d_end = torch.cuda.Event(enable_timing=True)

    for step in range(output_tokens):
        next_token = next_logits.argmax(dim=-1)

        if step == 0:
            d_start.record()
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        if step == 0:
            d_end.record()

        if step == 0:
            decode_raw = collector.collect_all()
            torch.cuda.synchronize()

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

    if decode_raw:
        decode_raw["_total_ms"] = d_start.elapsed_time(d_end)
    else:
        decode_raw["_total_ms"] = 0.0

    # Drain remaining decode events (steps 1..output_tokens-1) to avoid
    # contaminating the next iteration's prefill measurements.
    collector.drain()

    return prefill_raw, decode_raw


def _compute_fine_derived(agg: dict) -> dict:
    """Compute derived fine metrics.

    attn_core = full_attn_total - (q+k+v+o)   (the fused SDPA/FlashAttention kernel)
    hybrid_attn_core = hybrid_attn_total - (in_proj_* + out_proj + conv1d + act + norm)
      (the SSM recurrence core of GatedDeltaNet)
    """
    result = dict(agg)

    # Full attention core
    full_attn_total = result.pop("_full_attn_total", 0.0)
    q = result.get("q_proj", 0.0)
    k = result.get("k_proj", 0.0)
    v = result.get("v_proj", 0.0)
    o = result.get("o_proj", 0.0)
    result["attn_core"] = max(0.0, full_attn_total - (q + k + v + o))

    # Hybrid attention core (SSM recurrence)
    hybrid_total = result.pop("_hybrid_attn_total", 0.0)
    hybrid_tracked = (
        result.get("in_proj_qkv", 0.0)
        + result.get("in_proj_z", 0.0)
        + result.get("in_proj_a", 0.0)
        + result.get("in_proj_b", 0.0)
        + result.get("out_proj", 0.0)
        + result.get("hybrid_conv1d", 0.0)
        + result.get("activation", 0.0)  # only from hybrid layers (MLP act is additive)
    )
    # Note: activation is also tracked from MLP, so this is approximate.
    # We use a rough estimate: hybrid_attn_core = hybrid_total - linear - conv1d
    # The act/norm in hybrid are already counted in their own labels.
    result["hybrid_attn_core"] = max(0.0, hybrid_total - hybrid_tracked)

    # Compute all_linear
    result["all_linear"] = result.pop("_all_linear_internal", 0.0)

    return result


def _compute_other(pct_dict: dict, total_ms: float, mode: str) -> None:
    """Compute 'other' as 100% - sum of tracked percentages."""
    if mode == "coarse":
        tracked = ["hybrid_linear_attn_block", "full_attn_block", "mlp_block", "norm", "lm_head", "vision"]
    else:
        tracked = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
            "attn_core", "hybrid_attn_core", "hybrid_conv1d",
            "activation", "norm", "lm_head", "vision",
        ]
    tracked_sum = sum(pct_dict.get(k, 0.0) for k in tracked)
    pct_dict["other"] = max(0.0, 100.0 - tracked_sum)


def _average_timing_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    return {k: statistics.fmean([d.get(k, 0.0) for d in dicts]) for k in keys}


def _to_pct(raw: dict, total: float) -> dict:
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: v * 100.0 / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Config sweep
# ---------------------------------------------------------------------------


def run_config(
    model,
    processor,
    args,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    layer_info: dict,
) -> dict | None:
    """Run benchmark for a single config. Returns a dict row or None on OOM."""
    row = {
        "method": args.method,
        "model_variant": args.variant,
        "model_path": args.model_path,
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "multimodal": args.multimodal,
        "text_tokens": input_tokens,
        "image_tokens": 0,
        "total_prefill_tokens": input_tokens,
        "image_size": "",
        "dtype": str(args.dtype),
        "attn_implementation": args.attn_implementation,
        "input_mode": args.input_mode,
        "sampling": args.sampling,
        "checkpoint_path": args.checkpoint or "",
        **args.replacement_fields,
    }

    try:
        if args.multimodal:
            if processor is None:
                print("  Skipping multimodal: no processor loaded")
                return None
            inputs, mm_meta = prepare_multimodal_inputs(
                model, processor, batch_size, input_tokens,
                args.image_path, args.input_mode, args.prompt_text,
            )
            row.update({k: v if v is not None else "N/A" for k, v in mm_meta.items()})
            # For multimodal, we use the processor's output directly
            input_ids = inputs["input_ids"]
            # Remove input_ids from the dict to avoid duplicate; pass the rest as kwargs
            extra_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
            # Actually for simplicity, we wrap in a custom forward
            # For multimodal benchmark, we just use the model's generate-like forward
            # We wrap the model call to include pixel_values etc.
            # For now, let's benchmark the full multimodal forward by overriding
            return _run_multimodal_config(model, processor, args, batch_size, input_tokens, output_tokens,
                                         layer_info, row, inputs)
        else:
            input_ids = prepare_text_inputs(
                model, batch_size, input_tokens,
                mode=args.input_mode, prompt_text=args.prompt_text,
            )
            row["text_tokens"] = input_ids.shape[1]
            row["total_prefill_tokens"] = input_ids.shape[1]

        # --- Speed benchmark (no hooks) ---
        if not args.breakdown:
            speed = benchmark_prefill_decode(
                model, input_ids, output_tokens, args.warmup, args.iters, args.sampling
            )
            row.update(speed)
            return row

        # --- Breakdown benchmark (with hooks) ---
        breakdown = benchmark_breakdown(
            model, input_ids, output_tokens, args.warmup, args.iters,
            args.breakdown_mode, layer_info,
        )
        row.update(breakdown)
        return row

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  OOM: batch={batch_size} input={input_tokens} output={output_tokens}")
        row["status"] = "OOM"
        return row
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"  Error: batch={batch_size} input={input_tokens} output={output_tokens}: {e}")
        return None


def _run_multimodal_config(model, processor, args, batch_size, input_tokens, output_tokens,
                           layer_info, row, inputs):
    """Run multimodal benchmark. Passes all processor outputs (pixel_values,
    image_grid_thw, mm_token_type_ids, etc.) to the model forward.
    """
    device = next(model.parameters()).device

    # Extract input_ids and build extra kwargs from everything else the processor returned
    input_ids = inputs.pop("input_ids").to(device)
    extra_kwargs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            extra_kwargs[k] = v.to(device)
        else:
            extra_kwargs[k] = v

    row["text_tokens"] = input_ids.shape[1]
    row["total_prefill_tokens"] = input_ids.shape[1]

    def mm_forward(input_ids_batch, **cache_kwargs):
        # Only pass multimodal kwargs for prefill (no past_key_values).
        # During decode with KV cache, only input_ids and cache are needed.
        if cache_kwargs.get("past_key_values") is None:
            return model(input_ids=input_ids_batch, **extra_kwargs, **cache_kwargs)
        else:
            return model(input_ids=input_ids_batch, **cache_kwargs)

    if not args.breakdown:
        speed = _benchmark_with_forward_fn(
            mm_forward, input_ids, output_tokens, args.warmup, args.iters, args.sampling
        )
        row.update(speed)
        return row

    # Breakdown with multimodal
    collector = _HookTimingCollector()
    if args.breakdown_mode == "coarse":
        install_coarse_hooks(model, layer_info, collector)
    else:
        install_fine_hooks(model, layer_info, collector)

    # Warmup
    for _ in range(args.warmup):
        _do_one_breakdown_cycle_mm(model, input_ids, output_tokens, collector, extra_kwargs)

    prefill_results: list[dict[str, float]] = []
    decode_results: list[dict[str, float]] = []

    for _ in range(args.iters):
        pref_agg, dec_agg = _do_one_breakdown_cycle_mm(
            model, input_ids, output_tokens, collector, extra_kwargs
        )
        prefill_results.append(pref_agg)
        decode_results.append(dec_agg)

    prefill_avg = _average_timing_dicts(prefill_results)
    decode_avg = _average_timing_dicts(decode_results)

    if args.breakdown_mode == "fine":
        prefill_avg = _compute_fine_derived(prefill_avg)
        decode_avg = _compute_fine_derived(decode_avg)
    else:
        prefill_avg["all_linear"] = prefill_avg.pop("_all_linear_internal", 0.0)
        decode_avg["all_linear"] = decode_avg.pop("_all_linear_internal", 0.0)

    prefill_total = prefill_avg.get("_total_ms", sum(prefill_avg.values()))
    decode_total = decode_avg.get("_total_ms", sum(decode_avg.values()))

    prefill_pct = _to_pct(prefill_avg, prefill_total)
    decode_pct = _to_pct(decode_avg, decode_total)

    _compute_other(prefill_pct, prefill_total, args.breakdown_mode)
    _compute_other(decode_pct, decode_total, args.breakdown_mode)

    result = {}
    for k, v in prefill_pct.items():
        result[f"prefill_{k}_pct"] = v
    for k, v in decode_pct.items():
        result[f"decode_{k}_pct"] = v
    result["prefill_total_ms"] = prefill_total
    result["decode_total_ms"] = decode_total
    row.update(result)
    return row


def _benchmark_with_forward_fn(forward_fn, input_ids, output_tokens, warmup, iters, sampling):
    """Benchmark using a custom forward function (for multimodal)."""
    def cycle():
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        outputs = forward_fn(input_ids, use_cache=True)
        end_ev.record()
        torch.cuda.synchronize()
        prefill_ms = start_ev.elapsed_time(end_ev)

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

        decode_times: list[float] = []
        d_start = torch.cuda.Event(enable_timing=True)
        d_end = torch.cuda.Event(enable_timing=True)

        for _ in range(output_tokens):
            if sampling:
                next_token = torch.multinomial(
                    torch.softmax(next_logits.squeeze(1).float(), dim=-1), num_samples=1
                ).unsqueeze(0)
            else:
                next_token = next_logits.argmax(dim=-1)

            d_start.record()
            outputs = forward_fn(next_token, past_key_values=past_key_values, use_cache=True)
            d_end.record()
            torch.cuda.synchronize()
            decode_times.append(d_start.elapsed_time(d_end))

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1:, :]

        return prefill_ms, decode_times

    for _ in range(warmup):
        cycle()

    torch.cuda.synchronize()

    prefill_times = []
    all_decode_times = []
    first_decode_times = []

    for _ in range(iters):
        p_ms, d_list = cycle()
        prefill_times.append(p_ms)
        if d_list:
            first_decode_times.append(d_list[0])
            all_decode_times.extend(d_list[1:] if len(d_list) > 1 else [])

    mean_prefill_ms = statistics.fmean(prefill_times)
    mean_first = statistics.fmean(first_decode_times) if first_decode_times else 0.0
    mean_decode = statistics.fmean(all_decode_times) if all_decode_times else 0.0
    batch_size = input_ids.shape[0]
    tps = batch_size * 1000.0 / mean_decode if mean_decode > 0 else 0.0

    return {
        "prefill_ms": mean_prefill_ms,
        "decode_total_ms": mean_decode * (output_tokens - 1) if output_tokens > 1 else 0.0,
        "first_decode_ms": mean_first,
        "decode_per_token_ms": mean_decode,
        "tokens_per_sec": tps,
    }


def _do_one_breakdown_cycle_mm(model, input_ids, output_tokens, collector, extra_kwargs):
    """One breakdown cycle for multimodal model.

    extra_kwargs (pixel_values, image_grid_thw, etc.) are only passed during
    prefill. Decode steps use only input_ids and past_key_values.
    """
    p_start = torch.cuda.Event(enable_timing=True)
    p_end = torch.cuda.Event(enable_timing=True)

    p_start.record()
    outputs = model(input_ids=input_ids, **extra_kwargs, use_cache=True)
    p_end.record()

    prefill_raw = collector.collect_all()
    torch.cuda.synchronize()
    prefill_raw["_total_ms"] = p_start.elapsed_time(p_end)

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1:, :]

    d_start = torch.cuda.Event(enable_timing=True)
    d_end = torch.cuda.Event(enable_timing=True)
    decode_raw: dict[str, float] = {}

    for step in range(output_tokens):
        next_token = next_logits.argmax(dim=-1)
        if step == 0:
            d_start.record()
        # Decode: no multimodal kwargs, only cache
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        if step == 0:
            d_end.record()
        if step == 0:
            decode_raw = collector.collect_all()
            torch.cuda.synchronize()
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1:, :]

    if decode_raw:
        decode_raw["_total_ms"] = d_start.elapsed_time(d_end)
    else:
        decode_raw["_total_ms"] = 0.0

    collector.drain()

    return prefill_raw, decode_raw


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def get_csv_columns(args) -> list[str]:
    """Return the list of CSV column names based on benchmark mode."""
    base = [
        "method", "model_variant", "model_path", "batch_size", "input_tokens", "output_tokens",
        "multimodal", "text_tokens", "image_tokens", "total_prefill_tokens",
        "image_size", "dtype", "attn_implementation", "input_mode", "sampling", "checkpoint_path",
        "kernel_backend", "nvfp4_block_size", "nvfp4_backend", "nvfp4_quant_backend",
        "nvfp4_sf_layout", "marlin_activation_dtype", "replaced_linear_count",
        "skipped_linear_count", "compressed_module_count", "packed_checkpoint_file_size_bytes",
    ]
    if args.breakdown:
        # Breakdown mode
        prefixes = ["prefill_", "decode_"]
        labels = COARSE_MODULE_LABELS if args.breakdown_mode == "coarse" else FINE_MODULE_LABELS
        for prefix in prefixes:
            for label in labels:
                base.append(f"{prefix}{label}_pct")
        base.extend(["prefill_total_ms", "decode_total_ms", "status"])
    else:
        # No-hooks speed mode
        base.extend([
            "prefill_ms", "decode_total_ms", "first_decode_ms",
            "decode_per_token_ms", "tokens_per_sec", "status",
        ])
    return base


def write_csv_row(path: str, row: dict, columns: list[str], is_first: bool) -> None:
    """Write or append a row to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if is_first else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if is_first:
            writer.writeheader()
        writer.writerow(row)


def print_console_table(rows: list[dict], args) -> None:
    """Print a formatted console table of results."""
    if not rows:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("Qwen3.5 Speed Benchmark Results")
    print("=" * 100)

    if args.breakdown:
        print(f"\nBreakdown mode: {args.breakdown_mode}")
        for row in rows:
            status = row.get("status", "OK")
            if status == "OOM":
                print(f"  b={row['batch_size']} in={row['input_tokens']} out={row['output_tokens']}: OOM")
                continue
            print(f"\n  batch={row['batch_size']} input={row['input_tokens']} output={row['output_tokens']}")
            print(f"    Prefill total: {row.get('prefill_total_ms', 0):.2f}ms")
            print(f"    Decode total: {row.get('decode_total_ms', 0):.2f}ms")
            labels = COARSE_MODULE_LABELS if args.breakdown_mode == "coarse" else FINE_MODULE_LABELS
            print(f"    Prefill breakdown:")
            for label in labels:
                key = f"prefill_{label}_pct"
                if key in row and row[key] > 0.1:
                    print(f"      {label}: {row[key]:.1f}%")
            print(f"    Decode breakdown:")
            for label in labels:
                key = f"decode_{label}_pct"
                if key in row and row[key] > 0.1:
                    print(f"      {label}: {row[key]:.1f}%")
    else:
        print(f"\n{'Batch':>6} {'Input':>8} {'Output':>8} {'Prefill':>10} {'FirstDec':>10} {'Dec/tok':>10} {'tok/s':>10}")
        print("-" * 65)
        for row in rows:
            status = row.get("status", "OK")
            if status == "OOM":
                print(f"{row['batch_size']:>6} {row['input_tokens']:>8} {row['output_tokens']:>8} {'OOM':>10}")
                continue
            print(
                f"{row['batch_size']:>6} {row['input_tokens']:>8} {row['output_tokens']:>8} "
                f"{row['prefill_ms']:>8.1f}ms {row['first_decode_ms']:>8.1f}ms "
                f"{row['decode_per_token_ms']:>8.2f}ms {row['tokens_per_sec']:>8.1f}"
            )


def write_json_output(path: str, rows: list[dict], args, layer_info: dict | None = None) -> None:
    """Write detailed results as JSON."""
    output = {
        "config": {
            "method": args.method,
            "model_variant": args.variant,
            "model_path": args.model_path,
            "dtype": str(args.dtype),
            "attn_implementation": args.attn_implementation,
            "multimodal": args.multimodal,
            "input_mode": args.input_mode,
            "sampling": args.sampling,
        },
        "model_summary": {
            "num_full_attn": layer_info["num_full_attn"] if layer_info else 0,
            "num_hybrid_linear_attn": layer_info["num_hybrid_linear_attn"] if layer_info else 0,
            "num_mlp": layer_info["num_mlp"] if layer_info else 0,
            "num_vision_blocks": layer_info["num_vision_blocks"] if layer_info else 0,
        },
        "results": rows,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen3.5 prefill/decode speed benchmark with module-level breakdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TODO model sizes (pass via --model-path):
  /home/agent/wja/data/models/Qwen/Qwen3.5-2B
  /home/agent/wja/data/models/Qwen/Qwen3.5-4B
  /home/agent/wja/data/models/Qwen/Qwen3.5-9B
  /home/agent/wja/data/models/Qwen/Qwen3.5-27B

nsys cross-validation:
  nsys profile --trace=cuda,nvtx,cublas,cudnn -o qwen_profile \\
    python scripts/bench_qwen3_5_speed.py --input-tokens 1024 --output-tokens 128
""",
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to Qwen3.5 model")
    p.add_argument("--variant", default=DEFAULT_QWEN3_5_VARIANT, choices=QWEN3_5_VARIANTS,
                   help="Qwen3.5 variant under /home/agent/wja/data/models/Qwen")
    p.add_argument("--method", default="dense", choices=QWEN3_5_REAL_KERNEL_METHODS,
                   help="Runtime method")
    p.add_argument("--checkpoint", default=None, help="Packed checkpoint for compressed runtime methods")
    p.add_argument("--device-map", default=None,
                   help='Optional Transformers device_map, e.g. "auto" for large models')
    p.add_argument("--max-memory", nargs="*", default=None, metavar="DEVICE:MEM",
                   help='Optional max_memory entries, e.g. "0:30GiB" "1:30GiB"')
    p.add_argument("--input-tokens", type=int, nargs="+", default=DEFAULT_INPUT_TOKENS,
                   help="Input token lengths to test")
    p.add_argument("--output-tokens", type=int, nargs="+", default=DEFAULT_OUTPUT_TOKENS,
                   help="Output token lengths to test")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                   help="Batch sizes to test")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=20, help="Measurement iterations")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Model dtype")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="Attention backend")
    p.add_argument("--breakdown", action="store_true",
                   help="Enable hook-based module breakdown (separate run)")
    p.add_argument("--breakdown-mode", default="coarse", choices=["coarse", "fine"],
                   help="Breakdown granularity")
    p.add_argument("--multimodal", action="store_true",
                   help="Include image input (multimodal benchmark)")
    p.add_argument("--image-path", help="Path to test image for multimodal mode")
    p.add_argument("--sampling", action="store_true",
                   help="Use multinomial sampling instead of argmax")
    p.add_argument("--input-mode", default="random", choices=["random", "repeat", "file"],
                   help="Input token generation mode")
    p.add_argument("--prompt-text", help="Prompt text for repeat mode or file path for file mode")
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="CSV output path")
    p.add_argument("--output-json", help="JSON output path for detailed per-layer results")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed per-layer breakdown to console")
    return p.parse_args()


def _parse_max_memory(entries: list[str] | None) -> dict[int | str, str] | None:
    if not entries:
        return None
    result: dict[int | str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid --max-memory entry {entry!r}; expected DEVICE:MEM")
        device, memory = entry.split(":", 1)
        result[int(device) if device.isdigit() else device] = memory
    return result


def main():
    args = parse_args()
    default_path = str(qwen3_5_model_path(args.variant))
    if args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = default_path
    if args.output_csv == DEFAULT_OUTPUT_CSV:
        variant_key = args.variant.lower().replace(".", "_")
        args.output_csv = f"artifacts/results/qwen3_5_{variant_key}_{args.method}/speed.csv"
    if args.method != "dense" and args.checkpoint is None:
        args.checkpoint = default_qwen3_5_kernel_checkpoint_path(args.variant, args.method)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    if args.method != "dense" and dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("compressed runtime methods support only --dtype bf16 or fp16")
    if args.method != "dense" and args.multimodal:
        raise ValueError("compressed packed checkpoints currently support Qwen3.5 text-only CausalLM benchmarking")

    print(f"Loading model from {args.model_path}")
    print(f"  method={args.method}, variant={args.variant}, dtype={args.dtype}, "
          f"attn={args.attn_implementation}, multimodal={args.multimodal}, device_map={args.device_map}")

    max_memory = _parse_max_memory(args.max_memory)
    model = load_model(
        args.model_path,
        dtype,
        args.attn_implementation,
        args.multimodal,
        device_map=args.device_map,
        max_memory=max_memory,
    )
    checkpoint_device = "auto" if args.device_map is not None else "cuda"
    if args.method == "dense" and args.device_map is None:
        model = model.to("cuda")
    replacement_fields = {
        "kernel_backend": "torch_dense",
        "nvfp4_block_size": "",
        "nvfp4_backend": "",
        "nvfp4_quant_backend": "",
        "nvfp4_sf_layout": "",
        "marlin_activation_dtype": "",
        "replaced_linear_count": "",
        "skipped_linear_count": "",
        "compressed_module_count": "",
        "packed_checkpoint_file_size_bytes": "",
    }
    if args.method != "dense":
        checkpoint_metadata, report = load_qwen3_5_kernel_checkpoint_into_model(
            model,
            args.checkpoint,
            device=checkpoint_device,
        )
        replacement_fields = {
            **report.csv_fields(),
            "compressed_module_count": report.replaced_linear_count,
            "packed_checkpoint_file_size_bytes": checkpoint_metadata.get("packed_checkpoint_file_size_bytes", ""),
        }
        print(
            "Qwen3.5 compressed kernel checkpoint loaded: "
            f"replaced={report.replaced_linear_count} skipped={report.skipped_linear_count}"
        )
        if report.skipped:
            print(f"skipped_modules={report.skipped[:10]}")
    args.replacement_fields = replacement_fields

    processor = None
    if args.multimodal:
        if not args.image_path:
            print("ERROR: --image-path is required for multimodal mode")
            sys.exit(1)
        processor = load_processor(args.model_path)
        print(f"  Processor loaded: {type(processor).__name__}")

    # Classify and print summary
    layer_info = classify_layers(model)
    print_model_summary(model, layer_info)
    attn_backend = getattr(
        getattr(model.config, "text_config", model.config), "_attn_implementation", args.attn_implementation
    )
    print(f"Attention backend: {attn_backend}")
    print(f"Dtype: {dtype}")
    print()

    # Generate config sweep
    configs = [
        (bs, itok, otok)
        for bs in args.batch_sizes
        for itok in args.input_tokens
        for otok in args.output_tokens
    ]
    print(f"Total configs to test: {len(configs)}")
    print(f"Mode: {'breakdown (' + args.breakdown_mode + ')' if args.breakdown else 'speed (no hooks)'}")

    csv_columns = get_csv_columns(args)
    all_rows: list[dict] = []
    is_first_csv = True

    for idx, (bs, itok, otok) in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] batch={bs} input={itok} output={otok} ...", end=" ", flush=True)
        t0 = time.time()
        row = run_config(model, processor, args, bs, itok, otok, layer_info)
        elapsed = time.time() - t0
        if row is None:
            print(f"ERROR ({elapsed:.1f}s)")
            continue
        status = row.get("status", "OK")
        if status == "OOM":
            print(f"OOM ({elapsed:.1f}s)")
        else:
            if args.breakdown:
                print(f"prefill={row.get('prefill_total_ms', 0):.1f}ms decode={row.get('decode_total_ms', 0):.1f}ms ({elapsed:.1f}s)")
            else:
                print(f"prefill={row['prefill_ms']:.1f}ms decode/tok={row['decode_per_token_ms']:.2f}ms ({elapsed:.1f}s)")

        all_rows.append(row)
        write_csv_row(args.output_csv, row, csv_columns, is_first_csv)
        is_first_csv = False

    # Console summary
    print_console_table(all_rows, args)

    # JSON output
    if args.output_json:
        write_json_output(args.output_json, all_rows, args, layer_info)
        print(f"\nJSON output written to {args.output_json}")

    print(f"\nCSV output written to {args.output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
