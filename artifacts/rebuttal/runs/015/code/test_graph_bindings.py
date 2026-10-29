"""Exact old/new GEMM comparison with changing GPU alpha during graph replay."""
import torch

from common import RUN, write_json
import cutlass_wrapper.nvfp4 as dense
import cutlass_wrapper.sparse_nvfp4 as sparse
from cutlass_wrapper.canonical_nvfp4 import quantize_nvfp4_canonical_weight


@torch.inference_mode()
def main():
    original_dense, original_sparse = dense._load_ext(), sparse._load_ext()
    dense._load_ext.cache_clear()
    sparse._load_ext.cache_clear()
    from graph_bindings import install
    install()
    new_dense, new_sparse = dense._load_ext(), sparse._load_ext()
    torch.manual_seed(148016)
    rows = []
    for m, n, k in [(8, 6144, 4096), (64, 6144, 4096), (2048, 6144, 4096), (8, 24576, 4096)]:
        weight = torch.randn(n, k, dtype=torch.bfloat16, device='cuda') * .02
        activation = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        for method in ['dense', 'sparse']:
            if method == 'dense':
                w = quantize_nvfp4_canonical_weight(weight)
                packed = dense.canonical_to_cutlass_nvfp4_weight(w)
                a, scale, global_scale = dense.pack_nvfp4_a(activation)
                alpha = torch.reciprocal(global_scale * packed.global_scale).reshape(1)
                arguments = [a, scale, packed.packed_weight, packed.scale, alpha, m, n, k]
                old = lambda: original_dense.nvfp4_gemm_bf16(*arguments)
                new = lambda: new_dense.nvfp4_gemm_bf16(*arguments)
            else:
                packed = sparse.quantize_sparse_weight_bf16(weight, prune=True)
                padded = torch.nn.functional.pad(activation, (0, 0, 0, (-m) % 32))
                b, scale, global_scale = sparse.pack_sparse_nvfp4_b(padded)
                alpha = torch.reciprocal(global_scale * packed.global_scale).reshape(1)
                arguments = [n, packed.sparse_weight, packed.metadata, b, packed.scale, scale, alpha]
                old = lambda: original_sparse.sparse_nvf4_gemm_from_sparse_packed(*arguments)
                new = lambda: new_sparse.sparse_nvf4_gemm_from_sparse_packed(*arguments)
            new()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = new()
            base = alpha.clone()
            signatures = []
            errors = []
            for factor in [.25, 1., 4.]:
                alpha.copy_(base * factor)
                graph.replay()
                torch.cuda.synchronize()
                expected = old()
                error = float((result.float() - expected.float()).abs().max())
                assert error == 0., (method, m, n, k, factor, error)
                errors.append(error)
                signatures.append(float(result.float().abs().mean()))
            assert signatures[0] < signatures[1] < signatures[2], 'Graph ignored the changing device alpha.'
            rows.append({'method': method, 'm': m, 'n': n, 'k': k,
                         'alpha_factors': [.25, 1., 4.], 'max_absolute_errors': errors,
                         'mean_absolute_outputs': signatures})
            print(rows[-1], flush=True)
            del old, new, graph, result, expected, arguments, alpha, base
            torch.cuda.synchronize()
    write_json(RUN / 'results/graph_bindings_numerical_gate.json',
               {'passed': True, 'cases': rows, 'comparisons': len(rows) * 3,
                'scope': 'Original and new bindings receive identical packed tensors and changing CUDA alpha. Graph outputs must match exactly.'})


if __name__ == '__main__':
    main()
