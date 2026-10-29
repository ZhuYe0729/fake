"""Local binding-only changes; retain original sources and quantized arithmetic."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN.parents[3] / 'fake/kernels/cutlass/cutlass_wrapper/cutlass_wrapper/csrc'


def main():
    reference = RUN / 'reference/kernel_sources'
    output = RUN / 'code/csrc'
    reference.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    names = ['nvfp4_sm120.cu', 'sparse_nvfp4_sm120.cu', 'sparse_nvfp4_bindings.cpp']
    original = {}
    for name in names:
        path = reference / name
        if not path.exists():
            path.write_bytes((SOURCE / name).read_bytes())
        original[name] = path.read_text()
    dense = original[names[0]]
    begin = dense.index('float alpha_to_host_float(')
    end = dense.index('\n}\n', begin) + 3
    dense = dense[:begin] + dense[end:]
    dense = dense.replace('float alpha_value = alpha_to_host_float(alpha);',
        'TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == at::kFloat && alpha.numel() == 1, "alpha must be a CUDA float32 scalar");')
    dense = dense.replace('      alpha_value);',
        '      1.0f);\n  arguments.epilogue.thread.alpha_ptr = alpha.data_ptr<float>();')
    dense = dense.replace('gemm.initialize(arguments, workspace.data_ptr<uint8_t>())',
        'gemm.initialize(arguments, workspace.data_ptr<uint8_t>(), at::cuda::getCurrentCUDAStream())')
    (output / names[0]).write_text(dense)

    declaration = '''extern "C" int qutlass_sparse_nvf4_device_alpha15(
    int m, int n, int k, float const* alpha,
    void const* a, void const* e, void const* b, void const* sfa, void const* sfb,
    void* d, void* workspace, int64_t* workspace_bytes, cudaStream_t stream);
'''
    cpp = original[names[2]].replace('extern "C" int qutlass_sparse_nvf4_gemm_from_sparse(',
                                    declaration + '\nextern "C" int qutlass_sparse_nvf4_gemm_from_sparse(', 1)
    start = cpp.index('  int status = qutlass_sparse_nvf4_gemm_from_sparse(')
    end = cpp.index('  TORCH_CHECK(status == 0,', start)
    cpp = cpp[:start] + '''  int64_t workspace_bytes = 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  int status = qutlass_sparse_nvf4_device_alpha15(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha.data_ptr<float>(),
      a_sparse_packed.data_ptr(), metadata_e.data_ptr(), b_packed.data_ptr(),
      a_sf.data_ptr(), b_sf.data_ptr(), out.data_ptr(), nullptr, &workspace_bytes, stream);
  TORCH_CHECK(status == 0, "sparse workspace query failed: ", status);
  auto workspace = torch::empty({workspace_bytes}, out.options().dtype(torch::kUInt8));
  status = qutlass_sparse_nvf4_device_alpha15(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), alpha.data_ptr<float>(),
      a_sparse_packed.data_ptr(), metadata_e.data_ptr(), b_packed.data_ptr(),
      a_sf.data_ptr(), b_sf.data_ptr(), out.data_ptr(), workspace.data_ptr(), nullptr, stream);
''' + cpp[end:]
    (output / names[2]).write_text(cpp)
    cuda = original[names[1]]
    insertion = declaration.replace(';\n', '\n') + '''{
  auto arguments = args_from_raw_ptrs(m, n, k, 1.0f, 0.0f,
      reinterpret_cast<ElementA::DataType const*>(a),
      reinterpret_cast<ElementE const*>(e),
      reinterpret_cast<ElementB::DataType const*>(b),
      reinterpret_cast<ElementA::ScaleFactorType const*>(sfa),
      reinterpret_cast<ElementB::ScaleFactorType const*>(sfb),
      reinterpret_cast<ElementC const*>(d), reinterpret_cast<ElementD*>(d));
  arguments.epilogue.thread.alpha_ptr = alpha;
  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  if (workspace_bytes != nullptr) {
    *workspace_bytes = static_cast<int64_t>(Gemm::get_workspace_size(arguments));
    return 0;
  }
  status = gemm.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  return static_cast<int>(gemm.run(stream));
}

'''
    cuda = cuda.replace('int run_backend_from_sparse(', insertion + 'int run_backend_from_sparse(', 1)
    (output / names[1]).write_text(cuda)
    (RUN / 'configs/graph_bindings_provenance.json').write_text(json.dumps({
        'sources': {name: {'original_sha256': hashlib.sha256(original[name].encode()).hexdigest(),
                           'modified_sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()}
                    for name in names},
        'changes': ['Read the existing FP32 alpha through the CUTLASS device scalar pointer.',
                    'Use the current CUDA stream for initialization and execution.',
                    'Sparse graph path allocates workspace through torch, avoiding raw cudaMalloc/free and the single cached run state.'],
        'unchanged': 'Quantization, scales, weight layouts, GEMM tiles, accumulator/output types.',
        'build_root': '/tmp/daso_graph_bindings_015_20261026'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
