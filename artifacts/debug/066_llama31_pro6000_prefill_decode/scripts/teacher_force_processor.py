"""vLLM-V1 registered logits processor for teacher-forced decode scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (AdapterLogitsProcessor,
                                              RequestLogitsProcessor)


class _TeacherForceRequest:
    def __init__(self, target_ids: list[int], capture_path: str) -> None:
        self.target_ids = target_ids
        self.capture_path = Path(capture_path)
        self.step = 0

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self.step >= len(self.target_ids):
            return logits
        target = self.target_ids[self.step]
        logprob = float(torch.log_softmax(logits.float(), dim=-1)[target])
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        with self.capture_path.open("a") as handle:
            handle.write(json.dumps({"step": self.step, "target_id": target,
                                     "logprob": logprob,
                                     "output_length": len(output_ids)}) + "\n")
        value = logits[target].clone()
        logits.fill_(float("-inf"))
        logits[target] = value
        self.step += 1
        return logits


class TeacherForceBatchProcessor(AdapterLogitsProcessor):
    """Registers a per-request target sequence via SamplingParams.extra_args."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        extra: Optional[dict[str, Any]] = params.extra_args
        if not extra or "teacher_force_target_ids" not in extra:
            return None
        target_ids = extra["teacher_force_target_ids"]
        capture_path = extra.get("teacher_force_capture_path")
        if (not isinstance(target_ids, list) or not target_ids or
                not all(isinstance(token, int) for token in target_ids) or
                not isinstance(capture_path, str)):
            raise ValueError("invalid teacher-force extra_args")
        return _TeacherForceRequest(target_ids, capture_path)

