from __future__ import annotations

from pathlib import Path

import torch


def load_fakevlm_dense(checkpoint: str | Path, device: str = "cuda"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    source = str(checkpoint)
    model = LlavaForConditionalGeneration.from_pretrained(
        source, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    processor.vision_feature_select_strategy = None
    return model, processor

