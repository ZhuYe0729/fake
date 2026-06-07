# Main Hybrid Policy Retest

This directory contains single-backend, manual-policy, and predictor-policy retests for Llama-2-7B, Llama-3.1-8B, and Qwen3.5-9B under `prefill_only`, `normal_01`, and `normal_02`.

The final metric is full-model E2E latency. Linear-module aggregate latency is retained to explain policy choices.
See `ANALYSIS.md` for the E2E ranking and main observations.
