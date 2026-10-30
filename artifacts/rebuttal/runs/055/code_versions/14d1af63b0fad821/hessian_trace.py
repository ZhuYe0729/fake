"""HAWQ trace-weighted sensitivity using true calibration-NLL Hessian-vector products."""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from common import MODEL, RUN, write_json


def trace_probe(loss: torch.Tensor, weights: list[torch.Tensor],
                vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    first = torch.autograd.grad(loss, weights, create_graph=True)
    estimates = []
    for index, (gradient, weight, vector) in enumerate(zip(first, weights, vectors, strict=True)):
        # Probe the diagonal Hessian block explicitly. A joint HVP is unbiased
        # for the trace too, but cross-module terms caused excessive variance.
        second, = torch.autograd.grad((gradient * vector).sum(), weight,
                                     retain_graph=index + 1 < len(weights))
        estimates.append((second.float() * vector.float()).sum().detach())
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--probes', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--layers-per-group', type=int, default=2)
    parser.add_argument('--output', type=Path, default=RUN / 'results/hessian_trace.json')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    started = time.perf_counter()
    torch.manual_seed(1729)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    blocks = torch.load(RUN / 'data/calibration_128_2112.pt', weights_only=True)[:args.blocks, :args.sequence_length]
    records = []
    for start in range(0, len(model.model.layers), args.layers_per_group):
        named = [(name, parameter) for name, parameter in model.named_parameters()
                 if name.startswith('model.layers.') and start <= int(name.split('.')[2]) < start + args.layers_per_group
                 and parameter.ndim == 2]
        weights = [parameter for _, parameter in named]
        for weight in weights:
            weight.requires_grad_(True)
        estimates = [[] for _ in weights]
        for block_index, block in enumerate(blocks):
            tokens = block[None].to('cuda')
            for probe in range(args.probes):
                loss = model(tokens, labels=tokens, use_cache=False).loss
                vectors = [torch.empty_like(weight).bernoulli_(.5).mul_(2).sub_(1) for weight in weights]
                values = trace_probe(loss, weights, vectors)
                for samples, value in zip(estimates, values, strict=True):
                    samples.append(float(value))
                del loss, vectors, values
                print('HVP', start, 'block', block_index, 'probe', probe,
                      'peak_GiB', torch.cuda.max_memory_allocated() / 1024**3, flush=True)
        for (name, weight), samples in zip(named, estimates, strict=True):
            mean = sum(samples) / len(samples)
            records.append({'module': name.removesuffix('.weight'), 'numel': weight.numel(),
                            'trace_samples': samples, 'trace_mean': mean,
                            'mean_eigenvalue': mean / weight.numel()})
            weight.requires_grad_(False)
        write_json(args.output, {
            'complete': start + args.layers_per_group >= len(model.model.layers),
            'args': vars(args), 'loss': 'teacher-forced causal NLL on WikiText train',
            'estimator': 'explicit diagonal-block Hutchinson, independent Rademacher vectors',
            'dtype': 'bfloat16 model and derivatives, float32 trace accumulation',
            'elapsed_seconds': time.perf_counter() - started, 'records': records})
        gc.collect()
        torch.cuda.empty_cache()
    print('Completed Hessian trace', time.perf_counter() - started, flush=True)


if __name__ == '__main__':
    main()
