#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

extern "C" int qutlass_sparse_nvf4_gemm_from_sparse(
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    void const* sparse_a_device_ptr,
    void const* metadata_e_device_ptr,
    void const* b_device_ptr,
    void const* sfa_device_ptr,
    void const* sfb_device_ptr,
    void* d_device_ptr,
    void const* c_device_ptr,
    void* sfd_device_ptr);

extern "C" int qutlass_sparse_nvf4_pack_from_dense(
    int m,
    int n,
    int k,
    void const* dense_a_device_ptr,
    void* sparse_a_device_ptr,
    void* metadata_e_device_ptr,
    int apply_random_2to4_mask);

extern "C" int qutlass_sparse_nvf4_pack_b(
    int rows,
    int k,
    void const* bf16_input_device_ptr,
    void* packed_b_device_ptr,
    void* sfb_device_ptr,
    void const* global_scale_device_ptr,
    void* scale_float_device_ptr,
    cudaStream_t stream);

extern "C" int64_t qutlass_sparse_nvf4_packed_a_numel(int m, int n, int k);
extern "C" int64_t qutlass_sparse_nvf4_metadata_e_numel(int m, int n, int k);

namespace {

int64_t ceil_div_int64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

int64_t sparse_scale_storage_numel(int64_t rows, int64_t k) {
  int64_t logical_cols = ceil_div_int64(k, 32);
  int64_t padded_rows = ceil_div_int64(rows, 128) * 128;
  int64_t padded_cols = ceil_div_int64(logical_cols, 4) * 4;
  return padded_rows * padded_cols;
}

int64_t sparse_fp4_storage_numel(int64_t m, int64_t n, int64_t k) {
  int64_t numel = qutlass_sparse_nvf4_packed_a_numel(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  TORCH_CHECK(numel > 0, "Failed to query packed sparse A size");
  return numel;
}

int64_t sparse_metadata_storage_numel(int64_t m, int64_t n, int64_t k) {
  int64_t numel = qutlass_sparse_nvf4_metadata_e_numel(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  TORCH_CHECK(numel > 0, "Failed to query sparse metadata size");
  return numel;
}

std::tuple<torch::Tensor, torch::Tensor> sparse_nvf4_pack_a_from_dense(
    torch::Tensor const& a_packed,
    bool apply_random_2to4_mask) {
  torch::checkAllContiguous(
      "sparse_nvf4_pack_a_from_dense",
      {{a_packed, "a_packed", 0}});
  torch::checkDeviceType(
      "sparse_nvf4_pack_a_from_dense",
      {a_packed},
      at::DeviceType::CUDA);
  TORCH_CHECK(a_packed.scalar_type() == at::kByte, "a_packed must be uint8");
  TORCH_CHECK(a_packed.dim() == 2, "a_packed must be 2D [M, K/2]");

  int64_t m = a_packed.size(0);
  int64_t k = a_packed.size(1) * 2;
  int64_t n = 1;
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16 for NVFP4");

  auto device = a_packed.device();
  auto sparse_a = torch::empty(
      {sparse_fp4_storage_numel(m, n, k)},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));
  auto metadata_e = torch::empty(
      {sparse_metadata_storage_numel(m, n, k)},
      torch::TensorOptions().dtype(torch::kUInt8).device(device));

  const at::cuda::OptionalCUDAGuard device_guard(device);
  int status = qutlass_sparse_nvf4_pack_from_dense(
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      a_packed.data_ptr(),
      sparse_a.data_ptr(),
      metadata_e.data_ptr(),
      apply_random_2to4_mask ? 1 : 0);
  TORCH_CHECK(status == 0, "qutlass_sparse_nvf4_pack_from_dense failed with status ", status);

  return std::make_tuple(sparse_a, metadata_e);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sparse_nvf4_pack_b(torch::Tensor const& b_bf16) {
  torch::checkAllContiguous(
      "sparse_nvf4_pack_b",
      {{b_bf16, "b_bf16", 0}});
  torch::checkDeviceType(
      "sparse_nvf4_pack_b",
      {b_bf16},
      at::DeviceType::CUDA);
  TORCH_CHECK(b_bf16.scalar_type() == at::kBFloat16, "b_bf16 must be bfloat16");
  TORCH_CHECK(b_bf16.dim() == 2, "b_bf16 must be 2D [N, K]");

  int64_t rows = b_bf16.size(0);
  int64_t k = b_bf16.size(1);
  TORCH_CHECK(rows > 0, "N/tokens must be positive");
  TORCH_CHECK(k > 0, "K must be positive");
  TORCH_CHECK(k % 32 == 0, "K must be divisible by 32 for sparse NVFP4 B scale groups");
  TORCH_CHECK(rows % 32 == 0, "N/tokens must be divisible by 32 for sparse NVFP4 B");

  auto device = b_bf16.device();
  const at::cuda::OptionalCUDAGuard device_guard(device);
  auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  auto float_options = torch::TensorOptions().dtype(at::kFloat).device(device);
  auto sf_options = torch::TensorOptions().dtype(at::kFloat8_e4m3fn).device(device);

  auto finite_abs = torch::nan_to_num(b_bf16.to(at::kFloat), 0.0, 0.0, 0.0).abs();
  auto max_abs = finite_abs.max().clamp_min(1.0e-12);
  auto global_scale = torch::full({1}, 448.0 * 6.0, float_options) / max_abs;
  auto packed_b = torch::empty({rows, k / 2}, byte_options);
  auto sfb = torch::empty({sparse_scale_storage_numel(rows, k)}, sf_options);
  auto scale_float = torch::empty({rows * (k / 32)}, float_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(sfb.data_ptr(), 0, static_cast<size_t>(sfb.numel()), stream.stream()));

  int status = qutlass_sparse_nvf4_pack_b(
      static_cast<int>(rows),
      static_cast<int>(k),
      b_bf16.data_ptr(),
      packed_b.data_ptr(),
      sfb.data_ptr(),
      global_scale.data_ptr(),
      scale_float.data_ptr(),
      stream.stream());
  TORCH_CHECK(status == 0, "qutlass_sparse_nvf4_pack_b failed with status ", status);
  return std::make_tuple(packed_b, sfb, global_scale.contiguous());
}

torch::Tensor sparse_nvf4_gemm_from_sparse_packed(
    int64_t m,
    torch::Tensor const& a_sparse_packed,
    torch::Tensor const& metadata_e,
    torch::Tensor const& b_packed,
    torch::Tensor const& a_sf,
    torch::Tensor const& b_sf,
    torch::Tensor const& alpha) {
  torch::checkAllContiguous(
      "sparse_nvf4_gemm_from_sparse_packed",
      {{a_sparse_packed, "a_sparse_packed", 0},
       {metadata_e, "metadata_e", 1},
       {b_packed, "b_packed", 2},
       {a_sf, "a_sf", 3},
       {b_sf, "b_sf", 4},
       {alpha, "alpha", 5}});
  torch::checkDeviceType(
      "sparse_nvf4_gemm_from_sparse_packed",
      {a_sparse_packed, metadata_e, b_packed, a_sf, b_sf, alpha},
      at::DeviceType::CUDA);
  torch::checkAllSameGPU(
      "sparse_nvf4_gemm_from_sparse_packed",
      {{a_sparse_packed, "a_sparse_packed", 0},
       {metadata_e, "metadata_e", 1},
       {b_packed, "b_packed", 2},
       {a_sf, "a_sf", 3},
       {b_sf, "b_sf", 4},
       {alpha, "alpha", 5}});

  TORCH_CHECK(a_sparse_packed.scalar_type() == at::kByte, "a_sparse_packed must be uint8");
  TORCH_CHECK(metadata_e.scalar_type() == at::kByte, "metadata_e must be uint8");
  TORCH_CHECK(b_packed.scalar_type() == at::kByte, "b_packed must be uint8");
  TORCH_CHECK(a_sf.scalar_type() == at::kFloat8_e4m3fn, "a_sf must be float8_e4m3fn");
  TORCH_CHECK(b_sf.scalar_type() == at::kFloat8_e4m3fn, "b_sf must be float8_e4m3fn");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float32");
  TORCH_CHECK(alpha.numel() == 1, "alpha must contain exactly one element");
  TORCH_CHECK(b_packed.dim() == 2, "b_packed must be 2D [N, K/2]");

  int64_t n = b_packed.size(0);
  int64_t k = b_packed.size(1) * 2;
  TORCH_CHECK(m > 0, "m must be positive");
  TORCH_CHECK(k % 16 == 0, "K must be divisible by 16 for NVFP4");
  TORCH_CHECK(
      a_sf.numel() == sparse_scale_storage_numel(m, k),
      "a_sf must use sparse NVFP4 blocked layout (sf_vec_size=32); got ",
      a_sf.numel(),
      " elements, expected ",
      sparse_scale_storage_numel(m, k));
  TORCH_CHECK(
      b_sf.numel() == sparse_scale_storage_numel(n, k),
      "b_sf must use sparse NVFP4 blocked layout (sf_vec_size=32); got ",
      b_sf.numel(),
      " elements, expected ",
      sparse_scale_storage_numel(n, k));
  TORCH_CHECK(
      a_sparse_packed.numel() == sparse_fp4_storage_numel(m, n, k),
      "a_sparse_packed has unexpected storage size for the provided shape");
  TORCH_CHECK(
      metadata_e.numel() == sparse_metadata_storage_numel(m, n, k),
      "metadata_e has unexpected storage size for the provided shape");

  auto device = a_sparse_packed.device();
  auto out = torch::empty(
      {m, n},
      torch::TensorOptions().dtype(torch::kBFloat16).device(device));

  const at::cuda::OptionalCUDAGuard device_guard(device);
  int status = qutlass_sparse_nvf4_gemm_from_sparse(
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      alpha.item<float>(),
      0.0f,
      a_sparse_packed.data_ptr(),
      metadata_e.data_ptr(),
      b_packed.data_ptr(),
      a_sf.data_ptr(),
      b_sf.data_ptr(),
      out.data_ptr(),
      nullptr,
      nullptr);
  TORCH_CHECK(status == 0, "qutlass_sparse_nvf4_gemm_from_sparse failed with status ", status);
  return out;
}

torch::Tensor sparse_nvf4_gemm_from_sparse_packed_out(
    int64_t m,
    torch::Tensor const& a_sparse_packed,
    torch::Tensor const& metadata_e,
    torch::Tensor const& b_packed,
    torch::Tensor const& a_sf,
    torch::Tensor const& b_sf,
    torch::Tensor const& alpha,
    torch::Tensor const& out) {
  torch::checkAllContiguous(
      "sparse_nvf4_gemm_from_sparse_packed_out",
      {{a_sparse_packed, "a_sparse_packed", 0},
       {metadata_e, "metadata_e", 1},
       {b_packed, "b_packed", 2},
       {a_sf, "a_sf", 3},
       {b_sf, "b_sf", 4},
       {alpha, "alpha", 5},
       {out, "out", 6}});
  torch::checkDeviceType(
      "sparse_nvf4_gemm_from_sparse_packed_out",
      {a_sparse_packed, metadata_e, b_packed, a_sf, b_sf, alpha, out},
      at::DeviceType::CUDA);
  torch::checkAllSameGPU(
      "sparse_nvf4_gemm_from_sparse_packed_out",
      {{a_sparse_packed, "a_sparse_packed", 0},
       {metadata_e, "metadata_e", 1},
       {b_packed, "b_packed", 2},
       {a_sf, "a_sf", 3},
       {b_sf, "b_sf", 4},
       {alpha, "alpha", 5},
       {out, "out", 6}});

  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(out.dim() == 2, "out must be 2D [M, N]");

  int64_t n = b_packed.size(0);
  TORCH_CHECK(out.size(0) == m, "out has wrong M dimension");
  TORCH_CHECK(out.size(1) == n, "out has wrong N dimension");

  const at::cuda::OptionalCUDAGuard device_guard(a_sparse_packed.device());
  int status = qutlass_sparse_nvf4_gemm_from_sparse(
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(b_packed.size(1) * 2),
      alpha.item<float>(),
      0.0f,
      a_sparse_packed.data_ptr(),
      metadata_e.data_ptr(),
      b_packed.data_ptr(),
      a_sf.data_ptr(),
      b_sf.data_ptr(),
      out.data_ptr(),
      nullptr,
      nullptr);
  TORCH_CHECK(status == 0, "qutlass_sparse_nvf4_gemm_from_sparse failed with status ", status);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sparse_nvf4_pack_a_from_dense",
      &sparse_nvf4_pack_a_from_dense,
      "Pack a structurally sparse dense NVFP4 A operand into sparse A and metadata E.");
  m.def(
      "sparse_nvf4_pack_b",
      &sparse_nvf4_pack_b,
      "Pack a row-major BF16 activation matrix as sparse SM120 NVFP4 B + SFB.");
  m.def(
      "sparse_nvf4_gemm_from_sparse_packed",
      &sparse_nvf4_gemm_from_sparse_packed,
      "Run the SM120 sparse NVFP4 GEMM on pre-packed sparse A and metadata E.");
  m.def(
      "sparse_nvf4_gemm_from_sparse_packed_out",
      &sparse_nvf4_gemm_from_sparse_packed_out,
      "Run the SM120 sparse NVFP4 GEMM into a preallocated output tensor.");
}
