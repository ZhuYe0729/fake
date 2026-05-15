#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FlashInfer NVFP4 quantize + GEMM on CUDA.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--backend", choices=["auto", "cutlass", "cudnn", "trtllm", "cute-dsl", "b12x"], default="auto")
    parser.add_argument("--quant-backend", choices=["cuda", "cute-dsl"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    import flashinfer

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
    print(f"device={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)}")
    print(f"shape m={args.m} n={args.n} k={args.k} dtype={dtype} backend={args.backend}")

    a = torch.randn((args.m, args.k), device=device, dtype=dtype)
    b = torch.randn((args.n, args.k), device=device, dtype=dtype)
    b_shuffle = args.backend == "trtllm"
    a_global_sf, a_fp4, a_sf = _quantize_nvfp4(a, nvfp4_quantize, SfLayout, False, args.quant_backend)
    b_global_sf, b_fp4, b_sf = _quantize_nvfp4(b, nvfp4_quantize, SfLayout, b_shuffle, args.quant_backend)
    alpha = torch.reciprocal(a_global_sf * b_global_sf)
    out = mm_fp4(
        a_fp4,
        b_fp4.T,
        a_sf,
        b_sf.T,
        alpha,
        dtype,
        None,
        16,
        False,
        args.backend,
        True,
    )
    ref = a @ b.T
    diff = (out.float() - ref.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
    print(f"out_shape={tuple(out.shape)} out_dtype={out.dtype}")
    print(f"max_abs_err={diff.max().item():.6f} mean_abs_err={diff.mean().item():.6f} cosine={cosine:.6f}")

    nvfp4_mm_ms = _bench_cuda(
        lambda: mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, alpha, dtype, None, 16, False, args.backend, True),
        args.warmup,
        args.iters,
    )
    dense_mm_ms = _bench_cuda(lambda: a @ b.T, args.warmup, args.iters)
    a_scale_ms = _bench_cuda(lambda: _global_scale(a), args.warmup, args.iters)
    a_quant_ms = _bench_cuda(
        lambda: nvfp4_quantize(
            a,
            a_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            sf_vec_size=16,
            backend=args.quant_backend,
        ),
        args.warmup,
        args.iters,
    )
    a_scale_quant_ms = _bench_cuda(
        lambda: _quantize_nvfp4(a, nvfp4_quantize, SfLayout, False, args.quant_backend),
        args.warmup,
        args.iters,
    )
    b_quant_once_ms = _bench_cuda(
        lambda: _quantize_nvfp4(b, nvfp4_quantize, SfLayout, b_shuffle, args.quant_backend),
        args.warmup,
        args.iters,
    )

    def nvfp4_forward_like():
        a_sf_global, a_q, a_descale = _quantize_nvfp4(a, nvfp4_quantize, SfLayout, False, args.quant_backend)
        a_alpha = torch.reciprocal(a_sf_global * b_global_sf)
        return mm_fp4(a_q, b_fp4.T, a_descale, b_sf.T, a_alpha, dtype, None, 16, False, args.backend, True)

    nvfp4_forward_like_ms = _bench_cuda(nvfp4_forward_like, args.warmup, args.iters)
    print(f"dense_mm_ms={dense_mm_ms:.6f}")
    print(f"nvfp4_mm_only_ms={nvfp4_mm_ms:.6f}")
    print(f"activation_global_scale_ms={a_scale_ms:.6f}")
    print(f"activation_quant_only_ms={a_quant_ms:.6f}")
    print(f"activation_scale_plus_quant_ms={a_scale_quant_ms:.6f}")
    print(f"weight_scale_plus_quant_once_ms={b_quant_once_ms:.6f}")
    print(f"nvfp4_forward_like_ms={nvfp4_forward_like_ms:.6f}")
    print(f"mm_only_speedup_vs_dense={dense_mm_ms / nvfp4_mm_ms:.3f}")
    print(f"forward_like_speedup_vs_dense={dense_mm_ms / nvfp4_forward_like_ms:.3f}")


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.float().abs().nan_to_num().max().clamp(min=1e-12)
    return torch.tensor([448.0 * 6.0], device=x.device, dtype=torch.float32) / max_abs


def _quantize_nvfp4(x, nvfp4_quantize, sf_layout, do_shuffle: bool, backend: str):
    global_sf = _global_scale(x)
    x_fp4, x_sf = nvfp4_quantize(
        x,
        global_sf,
        sfLayout=sf_layout.layout_128x4,
        do_shuffle=do_shuffle,
        sf_vec_size=16,
        backend=backend,
    )
    return global_sf, x_fp4, x_sf


def _bench_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


if __name__ == "__main__":
    main()
