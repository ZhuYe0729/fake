#!/usr/bin/env python3
"""Artifact-local guards for protocol changes in the pinned vLLM checkout."""
from __future__ import annotations

from typing import Any


def force_v1_chunked_prefill_disabled() -> None:
    """Preserve the paper protocol despite vLLM 0.11 overriding False in V1.

    EngineArgs._set_default_args unconditionally enables chunked prefill for
    non-pooling V1 models in this checkout.  Patch only the current process;
    the shared vLLM checkout remains untouched.
    """
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_cospaq064_chunked_prefill_guard", False):
        return
    original = EngineArgs._set_default_args

    def guarded(self: Any, usage_context: Any, model_config: Any) -> None:
        original(self, usage_context, model_config)
        if model_config.runner_type != "pooling":
            self.enable_chunked_prefill = False

    EngineArgs._set_default_args = guarded
    EngineArgs._cospaq064_chunked_prefill_guard = True


def assert_chunked_prefill_disabled(model: Any) -> bool:
    """Fail if the constructed engine does not match the paper protocol."""
    engine = getattr(model, "llm_engine", None)
    if engine is None:
        engine = getattr(getattr(model, "model", None), "llm_engine", None)
    if engine is None:
        raise RuntimeError("cannot locate vLLM engine for protocol audit")
    enabled = engine.vllm_config.scheduler_config.enable_chunked_prefill
    if enabled is not False:
        raise RuntimeError(f"chunked prefill protocol violation: {enabled=}")
    return True
