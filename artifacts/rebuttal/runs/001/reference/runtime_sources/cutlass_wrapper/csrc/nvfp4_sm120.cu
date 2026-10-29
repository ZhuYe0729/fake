#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

namespace cutlass_wrapper_nvfp4_sm120 {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    AlignmentC,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    AlignmentA,
    ElementB,
    LayoutBTag,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr int kNvfp4GroupSize = 16;
constexpr float kFp4Max = 6.0f;
constexpr float kNvfp4GlobalScaleNumerator = 448.0f * kFp4Max;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be torch.uint8")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be torch.bfloat16")

inline void check_shape(int64_t value, char const* name) {
  TORCH_CHECK(value > 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              name, " must fit positive int32, got ", value);
}

__device__ uint8_t fp4_code_from_float_device(float v) {
  return cutlass::float_e2m1_t(v).storage;
}

template <class LayoutSF>
__device__ int64_t scale_layout_offset(LayoutSF const& layout_sf, int row, int k) {
  return static_cast<int64_t>(layout_sf(cute::make_coord(row, k, 0)));
}

template <class LayoutSF>
__global__ void compute_scales_kernel(
    at::BFloat16 const* input,
    uint8_t* scale_storage,
    float* scale_float,
    float const* global_scale,
    LayoutSF layout_sf,
    int rows,
    int cols,
    int groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * groups;
  if (idx >= total) {
    return;
  }

  int row = idx / groups;
  int group = idx - row * groups;
  int col_begin = group * kNvfp4GroupSize;
  float max_abs = 0.0f;
  #pragma unroll
  for (int i = 0; i < kNvfp4GroupSize; ++i) {
    int col = col_begin + i;
    if (col < cols) {
      float v = static_cast<float>(input[row * cols + col]);
      v = isfinite(v) ? v : 0.0f;
      max_abs = fmaxf(max_abs, fabsf(v));
    }
  }

  float scale = max_abs > 0.0f ? (*global_scale) * max_abs / kFp4Max : 1.0f;
  scale = fminf(fmaxf(scale, 1.0e-8f), 448.0f);
  cutlass::float_ue4m3_t sf(scale);
  float quantized_scale = static_cast<float>(sf);
  if (quantized_scale == 0.0f) {
    sf = cutlass::float_ue4m3_t(1.0e-8f);
    quantized_scale = static_cast<float>(sf);
  }

  int64_t sf_offset = scale_layout_offset(layout_sf, row, col_begin);
  scale_storage[sf_offset] = sf.storage;
  scale_float[idx] = quantized_scale;
}

template <class LayoutSF>
__global__ void copy_logical_scales_kernel(
    uint8_t const* logical_scale,
    uint8_t* scale_storage,
    LayoutSF layout_sf,
    int rows,
    int cols,
    int groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * groups;
  if (idx >= total) {
    return;
  }

  int row = idx / groups;
  int group = idx - row * groups;
  int col_begin = group * kNvfp4GroupSize;
  int64_t sf_offset = scale_layout_offset(layout_sf, row, col_begin);
  scale_storage[sf_offset] = logical_scale[idx];
}

template <bool ColumnMajorOutput>
__global__ void pack_data_kernel(
    at::BFloat16 const* input,
    uint8_t* packed,
    float const* scale_float,
    float const* global_scale,
    int rows,
    int cols,
    int groups,
    int64_t total_elements) {
  int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_bytes = (total_elements + 1) / 2;
  if (byte_idx >= total_bytes) {
    return;
  }

  uint8_t codes[2] = {0, 0};
  #pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    int64_t elem = byte_idx * 2 + lane;
    if (elem >= total_elements) {
      continue;
    }

    int row;
    int col;
    if constexpr (ColumnMajorOutput) {
      row = static_cast<int>(elem % rows);
      col = static_cast<int>(elem / rows);
    } else {
      row = static_cast<int>(elem / cols);
      col = static_cast<int>(elem - static_cast<int64_t>(row) * cols);
    }

    float scale = scale_float[row * groups + col / kNvfp4GroupSize];
    float v = static_cast<float>(input[row * cols + col]);
    v = isfinite(v) ? v : 0.0f;
    v = v * (*global_scale) / scale;
    v = fminf(fmaxf(v, -kFp4Max), kFp4Max);
    codes[lane] = fp4_code_from_float_device(v);
  }

  packed[byte_idx] = static_cast<uint8_t>((codes[1] << 4) | codes[0]);
}

template <class LayoutSF>
int64_t scale_storage_size(LayoutSF const& layout_sf) {
  return static_cast<int64_t>(cute::size(cute::filter_zeros(layout_sf)));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pack_nvfp4_impl(torch::Tensor input, bool column_major) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  CHECK_BF16(input);
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  c10::cuda::CUDAGuard device_guard(input.device());

  int64_t rows64 = input.size(0);
  int64_t cols64 = input.size(1);
  check_shape(rows64, "rows");
  check_shape(cols64, "cols");
  TORCH_CHECK(cols64 % kNvfp4GroupSize == 0, "cols must be divisible by 16 for NVFP4 scale groups");
  TORCH_CHECK(cols64 % AlignmentA == 0, "cols must be divisible by 32 for the SM120 NVFP4 kernel");
  if (column_major) {
    TORCH_CHECK(rows64 % AlignmentB == 0, "B rows/out_features must be divisible by 32 for column-major packed B");
  }

  int rows = static_cast<int>(rows64);
  int cols = static_cast<int>(cols64);
  int groups = cols / kNvfp4GroupSize;

  auto byte_options = input.options().dtype(at::kByte);
  auto float_options = input.options().dtype(at::kFloat);
  auto finite_abs = torch::nan_to_num(input.to(at::kFloat), 0.0, 0.0, 0.0).abs();
  auto max_abs = finite_abs.max().clamp_min(1.0e-12);
  auto global_scale = torch::full({1}, kNvfp4GlobalScaleNumerator, float_options) / max_abs;
  auto packed = torch::empty({(rows64 * cols64 + 1) / 2}, byte_options);
  auto scale_float = torch::empty({rows64 * groups}, float_options);

  int64_t scale_elems;
  if (column_major) {
    auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, rows, cols, 1));
    scale_elems = scale_storage_size(layout_sfb);
    auto scales = torch::empty({scale_elems}, byte_options);
    int threads = 256;
    int blocks = (rows * groups + threads - 1) / threads;
    compute_scales_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<at::BFloat16 const*>(input.data_ptr<at::BFloat16>()),
        scales.data_ptr<uint8_t>(),
        scale_float.data_ptr<float>(),
        global_scale.data_ptr<float>(),
        layout_sfb,
        rows,
        cols,
        groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t total_elements = rows64 * cols64;
    blocks = static_cast<int>(((total_elements + 1) / 2 + threads - 1) / threads);
    pack_data_kernel<false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<at::BFloat16 const*>(input.data_ptr<at::BFloat16>()),
        packed.data_ptr<uint8_t>(),
        scale_float.data_ptr<float>(),
        global_scale.data_ptr<float>(),
        rows,
        cols,
        groups,
        total_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed, scales, global_scale};
  }

  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, cols, 1));
  scale_elems = scale_storage_size(layout_sfa);
  auto scales = torch::empty({scale_elems}, byte_options);
  int threads = 256;
  int blocks = (rows * groups + threads - 1) / threads;
  compute_scales_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<at::BFloat16 const*>(input.data_ptr<at::BFloat16>()),
      scales.data_ptr<uint8_t>(),
      scale_float.data_ptr<float>(),
      global_scale.data_ptr<float>(),
      layout_sfa,
      rows,
      cols,
      groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int64_t total_elements = rows64 * cols64;
  blocks = static_cast<int>(((total_elements + 1) / 2 + threads - 1) / threads);
  pack_data_kernel<false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<at::BFloat16 const*>(input.data_ptr<at::BFloat16>()),
      packed.data_ptr<uint8_t>(),
      scale_float.data_ptr<float>(),
      global_scale.data_ptr<float>(),
      rows,
      cols,
      groups,
      total_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales, global_scale};
}

torch::Tensor canonical_scale_to_sfb(torch::Tensor logical_scale, int64_t n64, int64_t k64) {
  CHECK_CUDA(logical_scale);
  CHECK_CONTIGUOUS(logical_scale);
  CHECK_UINT8(logical_scale);
  TORCH_CHECK(logical_scale.dim() == 2, "logical_scale must have shape [N, K / 16]");
  check_shape(n64, "n");
  check_shape(k64, "k");
  TORCH_CHECK(k64 % kNvfp4GroupSize == 0, "k must be divisible by 16 for NVFP4 scale groups");
  TORCH_CHECK(k64 % AlignmentA == 0, "k must be divisible by 32 for the SM120 NVFP4 kernel");
  TORCH_CHECK(n64 % AlignmentB == 0, "n must be divisible by 32 for the SM120 NVFP4 kernel");
  TORCH_CHECK(logical_scale.size(0) == n64 && logical_scale.size(1) == k64 / kNvfp4GroupSize,
              "logical_scale shape mismatch for N=", n64, ", K=", k64);
  c10::cuda::CUDAGuard device_guard(logical_scale.device());

  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  int groups = k / kNvfp4GroupSize;
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, n, k, 1));
  int64_t scale_elems = scale_storage_size(layout_sfb);
  auto scales = torch::empty({scale_elems}, logical_scale.options());
  int threads = 256;
  int blocks = (n * groups + threads - 1) / threads;
  copy_logical_scales_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logical_scale.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(),
      layout_sfb,
      n,
      k,
      groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scales;
}

typename Gemm::Arguments make_gemm_arguments(
    uint8_t const* a_data,
    uint8_t const* b_data,
    uint8_t const* sfa,
    uint8_t const* sfb,
    ElementC const* c_data,
    ElementD* d_data,
    int m,
    int n,
    int k,
    float alpha) {
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
          reinterpret_cast<ElementA::DataType const*>(a_data),
          stride_A,
          reinterpret_cast<ElementB::DataType const*>(b_data),
          stride_B,
          reinterpret_cast<ElementA::ScaleFactorType const*>(sfa),
          layout_SFA,
          reinterpret_cast<ElementB::ScaleFactorType const*>(sfb),
          layout_SFB,
      },
      {
          {alpha, 0.0f},
          c_data,
          stride_C,
          d_data,
          stride_D,
      }};
}

void check_gemm_inputs(torch::Tensor a_data, torch::Tensor sfa, torch::Tensor b_data, torch::Tensor sfb,
                       int64_t m64, int64_t n64, int64_t k64) {
  CHECK_CUDA(a_data);
  CHECK_CUDA(sfa);
  CHECK_CUDA(b_data);
  CHECK_CUDA(sfb);
  CHECK_CONTIGUOUS(a_data);
  CHECK_CONTIGUOUS(sfa);
  CHECK_CONTIGUOUS(b_data);
  CHECK_CONTIGUOUS(sfb);
  CHECK_UINT8(a_data);
  CHECK_UINT8(sfa);
  CHECK_UINT8(b_data);
  CHECK_UINT8(sfb);
  TORCH_CHECK(a_data.device() == sfa.device() && a_data.device() == b_data.device() && a_data.device() == sfb.device(),
              "all inputs must be on the same CUDA device");
  check_shape(m64, "m");
  check_shape(n64, "n");
  check_shape(k64, "k");
  TORCH_CHECK(k64 % AlignmentA == 0, "k must be divisible by 32 for the SM120 NVFP4 kernel");
  TORCH_CHECK(n64 % AlignmentB == 0, "n must be divisible by 32 for the SM120 NVFP4 kernel");
  TORCH_CHECK(a_data.numel() >= (m64 * k64 + 1) / 2, "a_data is too small for m x k packed FP4");
  TORCH_CHECK(b_data.numel() >= (n64 * k64 + 1) / 2, "b_data is too small for n x k packed FP4");
}

float alpha_to_host_float(torch::Tensor alpha) {
  TORCH_CHECK(alpha.numel() == 1, "alpha must be a scalar tensor");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat || alpha.scalar_type() == at::kDouble,
              "alpha must be float32 or float64");
  return alpha.item<float>();
}

}  // namespace cutlass_wrapper_nvfp4_sm120

using namespace cutlass_wrapper_nvfp4_sm120;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pack_nvfp4_a(torch::Tensor input) {
  return pack_nvfp4_impl(input, false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pack_nvfp4_b(torch::Tensor input) {
  return pack_nvfp4_impl(input, true);
}

bool can_implement_nvfp4_gemm(int64_t m64, int64_t n64, int64_t k64) {
  check_shape(m64, "m");
  check_shape(n64, "n");
  check_shape(k64, "k");
  if (k64 % AlignmentA != 0 || n64 % AlignmentB != 0) {
    return false;
  }
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  Gemm gemm;
  auto arguments = make_gemm_arguments(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, 1.0f);
  return gemm.can_implement(arguments) == cutlass::Status::kSuccess;
}

torch::Tensor nvfp4_gemm_bf16(
    torch::Tensor a_data,
    torch::Tensor sfa,
    torch::Tensor b_data,
    torch::Tensor sfb,
    torch::Tensor alpha,
    int64_t m64,
    int64_t n64,
    int64_t k64) {
  check_gemm_inputs(a_data, sfa, b_data, sfb, m64, n64, k64);
  c10::cuda::CUDAGuard device_guard(a_data.device());
  float alpha_value = alpha_to_host_float(alpha);

  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  auto out = torch::empty({m64, n64}, a_data.options().dtype(at::kBFloat16));
  auto c = torch::zeros({m64, n64}, out.options());

  Gemm gemm;
  auto arguments = make_gemm_arguments(
      a_data.data_ptr<uint8_t>(),
      b_data.data_ptr<uint8_t>(),
      sfa.data_ptr<uint8_t>(),
      sfb.data_ptr<uint8_t>(),
      reinterpret_cast<ElementC const*>(c.data_ptr<at::BFloat16>()),
      reinterpret_cast<ElementD*>(out.data_ptr<at::BFloat16>()),
      m,
      n,
      k,
      alpha_value);

  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS NVFP4 GEMM cannot implement this problem, status=", int(status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, a_data.options().dtype(at::kByte));
  status = gemm.initialize(arguments, workspace.data_ptr<uint8_t>());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS NVFP4 GEMM initialize failed, status=", int(status));

  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS NVFP4 GEMM run failed, status=", int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_nvfp4_a", &pack_nvfp4_a, "Pack row-major BF16 input as NVFP4 A operand");
  m.def("pack_nvfp4_b", &pack_nvfp4_b, "Pack row-major BF16 weight as NVFP4 B operand");
  m.def("canonical_scale_to_sfb", &canonical_scale_to_sfb, "Pack logical NVFP4 weight scale as CUTLASS SFB layout");
  m.def("can_implement_nvfp4_gemm", &can_implement_nvfp4_gemm, "Check SM120 dense NVFP4 GEMM support");
  m.def("nvfp4_gemm_bf16", &nvfp4_gemm_bf16, "Run SM120 dense NVFP4 GEMM with BF16 output");
}
